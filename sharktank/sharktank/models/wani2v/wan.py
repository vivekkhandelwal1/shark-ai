# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from collections import defaultdict
from einops import rearrange
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention
from torch.utils.checkpoint import checkpoint
from model_lib.audio_compressor import AudioAligner

__all__ = ['WanModel']


def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()


@amp.autocast(enabled=False)
def rope_apply_audio(x, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    # freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # x - [b, l, n, d], where d // 2 == c

    # loop over samples
    output = []
    for i in range(x.shape[0]):
        l = x[i].shape[0]

        # precompute multipliers
        x_i = torch.view_as_complex(x[i].to(torch.float64).reshape(
            l, n, -1, 2))
        freqs_i = freqs[:l].reshape(l, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()



class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.bfloat16()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.bfloat16()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, x_kv=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x, x_kv=None):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            if x_kv is not None:
                k = self.norm_k(self.k(x_kv)).view(b, s, n, d)
                v = self.v(x_kv).view(b, s, n, d)
            else:
                k = self.norm_k(self.k(x)).view(b, s, n, d)
                v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x, x_kv=x_kv)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x

class WanAI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 local_audio_attention=False):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.local_audio_attention = local_audio_attention

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        if self.local_audio_attention:
            self.k_audio = nn.Linear(768, dim)
            self.v_audio = nn.Linear(768, dim)
        else:
            self.k_audio = nn.Linear(dim, dim)
            self.v_audio = nn.Linear(dim, dim)
        self.norm_k_audio = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, grid_sizes, freqs, context, context_lens, audio_context, face_masks=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)

        # compute attention
        k_audio = rearrange(self.norm_k_audio(self.k_audio(audio_context)), "b l (n d) -> b l n d", n=n, d=d) # b is actuall bf!, l is 32, (n d) is dim
        v_audio = rearrange(self.v_audio(audio_context), "b l (n d) -> b l n d", n=n, d=d)
        if self.local_audio_attention:
            F = grid_sizes[0][0]
            ## audio is (b f) 32 h d, means we need to convert q to (b f) hw h d
            assert k_audio.shape[1] == 32 and k_audio.shape[2] == self.num_heads, f"{k_audio.shape}"
            bf = v_audio.shape[0]
            q=rope_apply(q, grid_sizes, freqs)
            q=rearrange(q, "b (f a) n d -> (b f) a n d", f=F)
            assert q.shape[0] == bf, f"{q.shape} vs {v_audio.shape}"

            # print(f'using local attention!, q: {q.shape}, k: {k_audio.shape}, v: {v_audio.shape}')

        audio_x = flash_attention(q, k_audio, v_audio, k_lens=None)

        if self.local_audio_attention:
            audio_x = rearrange(audio_x, "(b f) a n d -> b (f a) n d", f=F)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        audio_x = audio_x.flatten(2)
        if face_masks is not None:
            audio_x = audio_x * face_masks
        x = x + audio_x
        x = self.o(x)
        return x



WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'ai2v_cross_attn': WanAI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 local_audio_attention=False
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        
        if local_audio_attention:
            assert cross_attn_type == "ai2v_cross_attn", "local attention only supported for ai2v"
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                num_heads,
                                                                (-1, -1),
                                                                qk_norm,
                                                                eps,
                                                                local_audio_attention=local_audio_attention
                                                                )
        else:
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                        num_heads,
                                                                        (-1, -1),
                                                                        qk_norm,
                                                                        eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_context=None,
        face_masks=None,
        x_kv=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.bfloat16
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.bfloat16

        # self-attention
        if x_kv is None:
            y = self.self_attn(
                self.norm1(x).bfloat16() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
                freqs)
        else:
            y = self.self_attn(
                self.norm1(x).bfloat16() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
                freqs, x_kv=self.norm1(x_kv).bfloat16() * (1 + e[1]) + e[0])
            
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).bfloat16() * (1 + e[4]) + e[3])
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                x = x + y * e[5]
            return x

                # cross-attention & ffn function
        def audio_cross_attn_ffn(x, context, context_lens, e, audio_context, face_masks=None):
            x = x + self.cross_attn(self.norm3(x), grid_sizes, freqs, context, context_lens, audio_context, face_masks=face_masks)
            y = self.ffn(self.norm2(x).bfloat16() * (1 + e[4]) + e[3])
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                x = x + y * e[5]
            return x

        if audio_context is not None:
            x = audio_cross_attn_ffn(x, context, context_lens, e, audio_context, face_masks)
        else:
            x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.bfloat16
        with amp.autocast(dtype=torch.bfloat16):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 audio_dim=1280,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 do_audio_concat=True,
                 do_audio_cross_attn=False,
                 use_hallo_audio_proj_in=False,
                 use_whisper_audio_proj_in=False,
                 do_face_only_cross_attn=False,
                 use_guidance_embedding = False,
                 use_double_guidance_embedding = False,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ai2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_guidance_embedding = use_guidance_embedding
        self.enable_teacache = False
        self.accumulated_rel_l1_distance = defaultdict(int)
        self.previous_modulated_input = {}
        self.previous_residual_cond = {}
        self.previous_residual_uncond = {}
        self.use_double_guidance_embedding = use_double_guidance_embedding
        self.do_audio_concat = do_audio_concat
        self.audio_in = AudioAligner(in_channels=audio_dim, out_channels=16) if do_audio_concat else None
        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        
        self.do_audio_cross_attn = do_audio_cross_attn
        self.do_face_only_cross_attn = do_face_only_cross_attn
        self.use_hallo_audio_proj_in = use_hallo_audio_proj_in
        if do_audio_cross_attn:
            if use_whisper_audio_proj_in:
                self.audio_embedding = nn.Sequential(
                nn.Linear(audio_dim, dim), nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim))
            elif use_hallo_audio_proj_in:
                self.audio_embedding = AudioProjModel(output_dim=768)
            else:
                ## this is just regular proj in
                self.audio_embedding = []
                current_dim = audio_dim
                compress_ratio = 2 ## 768 * 12 = 9216 -> 4096 -> 2048 -> 1280
                while current_dim >= dim:
                    self.audio_embedding.append(nn.Linear(current_dim, max(current_dim // compress_ratio, dim)))
                    self.audio_embedding.append(nn.ReLU())
                    current_dim = current_dim // compress_ratio
                
                self.audio_embedding.append(nn.Linear(dim, dim))

                self.audio_embedding = nn.Sequential(*self.audio_embedding)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if self.use_guidance_embedding:
            self.guidance_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.guidance_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        
        if self.use_double_guidance_embedding:
            self.second_guidance_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.second_guidance_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))


        # blocks
        cross_attn_type = f'{model_type}_cross_attn'
        self.cross_attn_type = cross_attn_type
        self.blocks = nn.ModuleList([
            ## assumes local_audio_attention is used when we use hallo audio in..
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, 
                              local_audio_attention=self.use_hallo_audio_proj_in)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # if model_type == 'i2v':
        if 'i2v' in model_type:
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False


    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        current_step,
        segment_idx,
        clip_fea=None,
        y=None,
        audio=None,
        raw_audio=None,
        face_masks=None, 
        slg_layers=False,
        guidance=None,
        is_uncond=False,
        audio_guidance=None,
        overlap_width=0
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if "i2v" in self.model_type:
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        if self.audio_in:
            assert audio is not None, f"cant have audio in but no audio provided."

            audio = self.audio_in(audio.transpose(-1, -2)) ##  audio enters layer as b l c, leaves as b c l (aligned to frames)
            ## we need to make it b c f -> c f h w, we assume f is FIXED
            bs, aud_c, aud_seq = audio.shape

            x = [torch.cat([u, aud.reshape([aud_c, aud_seq, 1, 1]).repeat(1, 1, u.shape[-2], u.shape[-1])], dim=0) for u, aud in zip(x, audio)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] ## list of b c f h w

        ## hacked to only use the first one
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x[:1]])
        x = [u.flatten(2).transpose(1, 2) for u in x] ## 1 l c
        seq_lens = torch.tensor([u.size(1) for u in x[:1]], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        ## here i will loop the x instead to maintain as a list form
        x = [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
        

        if not self.do_face_only_cross_attn:
            face_masks = None

        if face_masks is not None: ## will just have 1 face mask
            ## list of face_masks c f h w
            face_masks = [rearrange(fm, "c f h w -> (f h w) c").unsqueeze(0) for fm in face_masks]
            face_masks = torch.cat([
                        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in face_masks
            ])
            # assert x.shape[:2] == face_masks.shape[:2], f"why is x and face mask shape diff? {x.shape} vs {face_masks.shape}"

        # time embeddings
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).bfloat16())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

            if self.use_guidance_embedding and guidance is not None:
                e_guidance = self.guidance_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, guidance * 1000.0).bfloat16())
                e0_guidance = self.guidance_projection(e_guidance).unflatten(1, (6, self.dim))
            
                e0 = e0 + e0_guidance
            
            if self.use_double_guidance_embedding and audio_guidance is not None:
                e_audio_guidance = self.second_guidance_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, audio_guidance * 1000.0).bfloat16())
                e0_audio_guidance = self.second_guidance_projection(e_audio_guidance).unflatten(1, (6, self.dim))
            
                e0 = e0 + e0_audio_guidance


        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        audio_contexts = None
        if self.do_audio_cross_attn:
            assert raw_audio is not None, f"Cant do cross attn with audio if no audio"
            assert self.cross_attn_type == "ai2v_cross_attn", f"Cant do cross attn if cross attn block is not ai2v"
            audio_contexts = [self.audio_embedding(raw_a) for raw_a in raw_audio] # B F dim or B F(21) 32 dim if hallo audio in
            if self.use_hallo_audio_proj_in:
                #print(f'rearranging after hallo audio in! ')
                audio_contexts = [rearrange(audio_context, "b f m c -> (b f) m c") for audio_context in audio_contexts]

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        should_calc = True
        if self.enable_teacache:
            if is_uncond:
                # If this is unconditional gen, then should always calculate this step if the corresponding step for cond gen is not skipped
                should_calc = self.should_calc
            else:
                if current_step <= self.teacache_start_step or current_step == self.num_steps - 1:
                    print("Current values:", current_step, self.teacache_start_step, self.num_steps)
                    should_calc = True
                    self.accumulated_rel_l1_distance[segment_idx] = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    temb_relative_l1 = self.relative_l1_distance(self.previous_modulated_input[segment_idx], e)
                    self.accumulated_rel_l1_distance[segment_idx] += rescale_func(((e-self.previous_modulated_input[segment_idx]).abs().mean() / self.previous_modulated_input[segment_idx].abs().mean()).cpu().item())
                    # self.accumulated_rel_l1_distance += temb_relative_l1
                    if self.accumulated_rel_l1_distance[segment_idx] < self.rel_l1_thresh:
                        should_calc = False
                        self.teacache_skipped_steps[segment_idx] += 1
                        print(f"Skipping step {current_step} due to teacache")
                    else:
                        print(f"Diff {self.accumulated_rel_l1_distance[segment_idx]}")
                        should_calc = True
                        self.accumulated_rel_l1_distance[segment_idx] = 0
                # self.previous_modulated_input = e
                self.previous_modulated_input[segment_idx] = e
                self.should_calc = should_calc

        if not should_calc:
            x += self.previous_residual_uncond[segment_idx] if is_uncond else self.previous_residual_cond[segment_idx]
        else:
            if self.enable_teacache:
                if is_uncond:
                    self.previous_residual_uncond[segment_idx] = None
                if not is_uncond:
                    self.previous_residual_cond[segment_idx] = None
                ori_hidden_states = x.clone()
            

            ## here is where we have to do the loopy
            """
            only things to change is: 
            x is a list of tensors
            audio_contexts is a list of tensors
            """
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens)
            
            if face_masks is not None:
                kwargs['face_masks'] = face_masks

            ## HERE IS WHERE WE BLEND
            for idx, block in enumerate(self.blocks):
                if slg_layers and idx == 9:
                    print(f"WARNING SKipping layer: {idx}")
                    continue
                # if False:
                if overlap_width > 0:
                    ## do blending and then do loop over segments
                    x_kvs = []
                    assert seq_len % 21 == 0, f"why isnt your latent frames divisible... {seq_len}"
                    num_to_change = seq_len // 21 * overlap_width ## this is the num video tokens to cache
                    for seg_idx in range(len(x)):
                        if seg_idx < len(x) - 1 and current_step < 1:
                            temp = x[seg_idx].clone()
                            temp[:, -num_to_change:] = x[seg_idx + 1][:, :num_to_change].clone()
                            x_kvs.append(temp)
                        else:
                            x_kvs.append(x[seg_idx].clone())
                            
                    x = [block(x=x[seg_idx], x_kv=x_kvs[seg_idx], audio_context=audio_contexts[seg_idx], **kwargs) for seg_idx in range(len(x))]
                else:
                    x = block(x=x[0], audio_context=audio_contexts[0], **kwargs)

            if self.enable_teacache:
                residual = ori_hidden_states 
                torch.sub(x, ori_hidden_states, out=residual)
                if is_uncond:
                    self.previous_residual_uncond[segment_idx] = residual
                if not is_uncond:
                    self.previous_residual_cond[segment_idx] = residual
                residual, ori_hidden_states = None, None

        # head
        x = [self.head(z, e) for z in x]

        # unpatchify
        x = [self.unpatchify([z], grid_sizes)[0] for z in x] ## list of b c f h w

        return [u.squeeze(0).bfloat16() for u in x]

    def relative_l1_distance(self, last_tensor, current_tensor):
        l1_distance = torch.abs(last_tensor - current_tensor).mean()
        norm = torch.abs(last_tensor).mean()
        relative_l1_distance = l1_distance / norm
        return relative_l1_distance.to(torch.float32)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def freeze_non_audio_params(self):
        num_audio_trainable = 0
        num_total = 0
        for n, p in self.named_parameters():
            if "audio" in n:
                p.requires_grad = True
                num_audio_trainable += p.numel()
            else:
                p.requires_grad = False
            num_total += p.numel()
        
        print(f"Training audio parameters only! {num_audio_trainable} / {num_total} requires grad")

    def freeze_audio_params(self):
        num_audio_trainable = 0
        num_total = 0
        for n, p in self.named_parameters():
            if "audio" not in n:
                p.requires_grad = True
                num_audio_trainable += p.numel()
            else:
                p.requires_grad = False
            num_total += p.numel()
        
        print(f"Training non audio parameters only! {num_audio_trainable} / {num_total} requires grad")



class AudioProjModel(torch.nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = torch.nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = torch.nn.LayerNorm(output_dim)
        
        self.conv1 = torch.nn.Conv1d(in_channels=context_tokens * output_dim,
                                     out_channels=context_tokens * output_dim,
                                     kernel_size=2,
                                     stride=2,
                                     padding=0)

    def forward(self, audio_embeds):
        # merge
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels) # (bz f) (w b c)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        # context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f (m c)", f=video_length
        )
        
        b, f, c = context_tokens.shape
        for _ in range(2):
            context_tokens = context_tokens.permute(0, 2, 1)
            if context_tokens.shape[-1] % 2 == 1:
                x_first, x_rest = context_tokens[..., 0], context_tokens[..., 1:]
                if x_rest.shape[-1] > 0:
                    x_rest = self.conv1(x_rest)

                context_tokens = torch.cat([x_first[..., None], x_rest], dim=-1)
                context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
            else:
                context_tokens = self.conv1(context_tokens)
                context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
        
        context_tokens = rearrange(context_tokens, "b f (m c) -> b f m c", m=self.context_tokens) 
        context_tokens = self.norm(context_tokens)       

        return context_tokens