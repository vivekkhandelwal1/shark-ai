# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
from dataclasses import dataclass
import math
from pathlib import Path
import torch
from typing import Callable

from einops import rearrange

from iree.compiler.ir import Context
from iree.turbine.aot import *
from iree.turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)

from transformers import CLIPTextModel
from sharktank.models.clip import ClipTextModel, ClipTextConfig
from sharktank.models.t5 import T5Encoder, T5Config
from sharktank.models.flux.flux import FluxModelV1, FluxParams, MLPEmbedder, LastLayer, timestep_embedding, EmbedND
from sharktank.models.vae.model import VaeDecoderModel
from sharktank.types.theta import Theta, Dataset, torch_module_to_theta
from sharktank.layers.mmdit import MMDITDoubleBlock, MMDITSingleBlock
from sharktank.layers.linear import LinearLayer


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float
    height: int
    width: int


@dataclass
class ModelSpec:
    ae_params: AutoEncoderParams
    ae_path: str | None


fluxconfigs = {
    "flux-dev": ModelSpec(
        ae_path=None,  # os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            height=1024,
            width=1024,
        ),
    ),
    "flux-schnell": ModelSpec(
        ae_path=None,  # os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            height=1024,
            width=1024,
        ),
    ),
}

model_repo_map = {
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}
model_file_map = {
    "flux-dev": "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors",
    "flux-schnell": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/flux1-schnell.safetensors",
}

torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def create_safe_name(hf_model_name, model_name_str=""):
    if not model_name_str:
        model_name_str = ""
    if model_name_str != "" and (not model_name_str.startswith("_")):
        model_name_str = "_" + model_name_str

    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub("\.", "_", safe_name)
    return safe_name


class FluxDenoiseStepModel(torch.nn.Module):
    def __init__(
        self,
        theta,
        params,
        batch_size=1,
        max_length=512,
        height=1024,
        width=1024,
    ):
        super().__init__()
        self.mmdit = FluxModelV1(theta=theta, params=params)
        self.batch_size = batch_size
        img_ids = torch.zeros(height // 16, width // 16, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 16)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 16)[None, :]
        self.img_ids = img_ids.reshape(1, height * width // 256, 3)
        self.txt_ids = torch.zeros(1, max_length, 3)

    def forward(self, img, txt, vec, step, timesteps, guidance_scale):
        guidance_vec = guidance_scale.repeat(self.batch_size)
        t_curr = torch.index_select(timesteps, 0, step)
        t_prev = torch.index_select(timesteps, 0, step + 1)
        t_vec = t_curr.repeat(self.batch_size)

        pred = self.mmdit(
            img=img,
            img_ids=self.img_ids,
            txt=txt,
            txt_ids=self.txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        # TODO: Use guidance scale
        # pred_uncond, pred = torch.chunk(pred, 2, dim=0)
        # pred = pred_uncond + guidance_scale * (pred - pred_uncond)
        img = img + (t_prev - t_curr) * pred
        return img


class FluxFrontDenoiseStepModel(torch.nn.Module):
    """First half of the denoising step model that processes up to the end of double blocks.
    
    This model handles the input processing and double blocks, stopping at the end of the 
    last double block.
    """
    def __init__(
        self,
        theta,
        params,
        batch_size=1,
        max_length=512,
        height=1024,
        width=1024,
    ):
        super().__init__()
        self.params = params
        self.batch_size = batch_size
        
        # Set up position embeddings
        img_ids = torch.zeros(height // 16, width // 16, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 16)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 16)[None, :]
        self.img_ids = img_ids.reshape(1, height * width // 256, 3)
        self.txt_ids = torch.zeros(1, max_length, 3)
        
        # Create position embedder
        self.pe_embedder = EmbedND(
            dim=params.hidden_size // params.num_heads, 
            theta=params.theta, 
            axes_dim=params.axes_dim
        )
        
        # Input processing components
        self.img_in = LinearLayer(theta("img_in"))
        self.time_in = MLPEmbedder(theta("time_in"))
        self.vector_in = MLPEmbedder(theta("vector_in"))
        self.guidance = params.guidance_embed
        if params.guidance_embed:
            self.guidance_in = MLPEmbedder(theta("guidance_in"))
        self.txt_in = LinearLayer(theta("txt_in"))
        
        # All double blocks
        self.double_blocks = torch.nn.ModuleList(
            [
                MMDITDoubleBlock(
                    theta("double_blocks", i),
                    num_heads=params.num_heads,
                    hidden_size=params.hidden_size,
                )
                for i in range(params.depth)
            ]
        )

    def forward(self, img, txt, vec, step, timesteps, guidance_scale):
        """Process the front half of the model up to the end of double blocks.
        
        Args:
            img: Input image latents
            txt: Input text embeddings
            vec: Input CLIP features
            step: Current denoising step
            timesteps: Timestep schedule
            guidance_scale: CFG guidance scale
            
        Returns:
            Intermediate state at the end of double blocks:
            (img_processed, txt_processed, vec_processed, t_curr)
        """
        # Initial processing
        guidance_vec = guidance_scale.repeat(self.batch_size)
        t_curr = torch.index_select(timesteps, 0, step)
        t_vec = t_curr.repeat(self.batch_size)
        
        # Process inputs
        img_processed = self.img_in(img)
        vec_processed = self.time_in(timestep_embedding(t_vec, 256))
        
        if self.guidance:
            vec_processed = vec_processed + self.guidance_in(
                timestep_embedding(guidance_vec, 256)
            )
            
        vec_processed = vec_processed + self.vector_in(vec)
        txt_processed = self.txt_in(txt)
        
        # Process through double blocks
        ids = torch.cat((self.txt_ids, self.img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        for block in self.double_blocks:
            img_processed, txt_processed = block(
                img=img_processed, txt=txt_processed, vec=vec_processed, pe=pe
            )
                
        # Return intermediate state at the end of double blocks
        return img_processed, txt_processed, vec_processed, t_curr


class FluxBackDenoiseStepModel(torch.nn.Module):
    """Second half of the denoising step model that processes from after double blocks to output.
    
    This model handles all single blocks and the final layer, starting from where the front model
    left off (after all double blocks).
    """
    def __init__(
        self,
        theta,
        params,
        batch_size=1,
        max_length=512,
        height=1024,
        width=1024,
    ):
        super().__init__()
        self.params = params
        self.batch_size = batch_size
        
        # Set up position embeddings
        img_ids = torch.zeros(height // 16, width // 16, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 16)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 16)[None, :]
        self.img_ids = img_ids.reshape(1, height * width // 256, 3)
        self.txt_ids = torch.zeros(1, max_length, 3)
        
        # Create position embedder
        self.pe_embedder = EmbedND(
            dim=params.hidden_size // params.num_heads, 
            theta=params.theta, 
            axes_dim=params.axes_dim
        )
        
        # All single blocks
        self.single_blocks = torch.nn.ModuleList(
            [
                MMDITSingleBlock(
                    theta("single_blocks", i),
                    num_heads=params.num_heads,
                    hidden_size=params.hidden_size,
                )
                for i in range(params.depth_single_blocks)
            ]
        )
        
        # Final layer
        self.final_layer = LastLayer(theta("final_layer"))

    def forward(self, img_inter, txt_inter, t_vec, t_curr, img, step, timesteps):
        """Process the back half of the model from after double blocks to output.
        
        Args:
            img_inter: Intermediate image state from front model
            txt_inter: Intermediate text state from front model
            t_vec: Processed timestep vector from front model
            t_curr: Current timestep from front model
            img: Original input image (needed for final update)
            step: Current denoising step
            timesteps: Timestep schedule
            
        Returns:
            Final updated image
        """
        # Process through blocks
        ids = torch.cat((self.txt_ids, self.img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        # Combine image and text for single blocks
        combined = torch.cat((txt_inter, img_inter), 1)
        
        # Process through single blocks
        for block in self.single_blocks:
            combined = block(combined, vec=t_vec, pe=pe)
        
        # Extract image part for final layer
        img_processed = combined[:, txt_inter.shape[1]:, ...]
        
        # Final projection
        pred = self.final_layer(img_processed, t_vec)
        
        # Apply timestep scaling
        t_prev = torch.index_select(timesteps, 0, step + 1)
        updated_img = img + (t_prev - t_curr) * pred
        
        return updated_img


@torch.no_grad()
def get_flux_transformer_model(
    weight_file,
    img_height=1024,
    img_width=1024,
    compression_factor=8,
    max_len=512,
    precision="fp32",
    bs=1,
    split_model=False,
):
    # DNS: refactor file to select datatype
    dtype = torch_dtypes[precision]
    transformer_dataset = Dataset.load(weight_file)
    flux_params = FluxParams.from_hugging_face_properties(transformer_dataset.properties)
    
    # Model configuration is available in flux_params
    
    if not split_model:
        # Create regular model
        model = FluxDenoiseStepModel(
            theta=transformer_dataset.root_theta,
            params=flux_params,
        )
        sample_args, sample_kwargs = model.mmdit.sample_inputs()
        sample_inputs = (
            sample_kwargs["img"],
            sample_kwargs["txt"],
            sample_kwargs["y"],
            torch.full((bs,), 1, dtype=torch.int64),
            torch.full((100,), 1, dtype=dtype),  # TODO: non-dev timestep sizes
            sample_kwargs["guidance"],
        )
        return model, sample_inputs
    else:
        # Create front half model (processes up to the end of double blocks)
        front_model = FluxFrontDenoiseStepModel(
            theta=transformer_dataset.root_theta,
            params=flux_params,
            batch_size=bs,
            max_length=max_len,
            height=img_height,
            width=img_width,
        )
        
        # Create back half model (processes from after double blocks to output)
        back_model = FluxBackDenoiseStepModel(
            theta=transformer_dataset.root_theta,
            params=flux_params,
            batch_size=bs,
            max_length=max_len,
            height=img_height,
            width=img_width,
        )
        
        # Prepare sample inputs
        flux_model = FluxModelV1(theta=transformer_dataset.root_theta, params=flux_params)
        sample_args, sample_kwargs = flux_model.sample_inputs()
        
        # Sample inputs are generated by the FluxModelV1.sample_inputs() method
        
        # Front model inputs (same as the full model)
        front_inputs = (
            sample_kwargs["img"],
            sample_kwargs["txt"],
            sample_kwargs["y"],
            torch.full((bs,), 1, dtype=torch.int64),
            torch.full((100,), 1, dtype=dtype),
            sample_kwargs["guidance"],
        )
        
        # Create placeholders with the correct shapes based on the model configuration
        hidden_size = flux_params.hidden_size
        
        # Create placeholder tensors with the correct shapes
        img_shape = list(sample_kwargs["img"].shape)
        img_shape[-1] = hidden_size
        img_inter_placeholder = torch.zeros(img_shape, dtype=dtype)
        
        txt_shape = list(sample_kwargs["txt"].shape)
        txt_shape[-1] = hidden_size
        txt_inter_placeholder = torch.zeros(txt_shape, dtype=dtype)
        
        t_vec_placeholder = torch.zeros((bs, hidden_size), dtype=dtype)
        t_curr_placeholder = torch.zeros((bs,), dtype=dtype)
        
        # Back model inputs with correct shapes
        back_inputs = (
            img_inter_placeholder,    # placeholder for img_inter with correct shape
            txt_inter_placeholder,    # placeholder for txt_inter with correct shape
            t_vec_placeholder,        # placeholder for t_vec with correct shape
            t_curr_placeholder,       # placeholder for t_curr with correct shape
            sample_kwargs["img"],     # original img
            torch.full((bs,), 1, dtype=torch.int64),  # step
            torch.full((100,), 1, dtype=dtype),       # timesteps
        )
            
        return (front_model, back_model), (front_inputs, back_inputs)


def get_flux_model_and_inputs(
    weight_file, precision, batch_size, max_length, height, width
):
    return get_flux_transformer_model(
        weight_file, height, width, 8, max_length, precision, batch_size
    )


# Copied from https://github.com/black-forest-labs/flux
class HFEmbedder(torch.nn.Module):
    def __init__(self, version: str, max_length: int, weight_file: str, **hf_kwargs):
        super().__init__()
        self.is_clip = version == "clip"
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            clip_dataset = Dataset.load(weight_file)
            config = ClipTextConfig.from_properties(clip_dataset.properties)
            self.hf_module = ClipTextModel(theta=clip_dataset.root_theta, config=config)
        else:
            t5_dataset = Dataset.load(weight_file)
            t5_config = T5Config.from_gguf_properties(
                t5_dataset.properties,
                feed_forward_proj="gated-gelu",
            )
            self.hf_module = T5Encoder(theta=t5_dataset.root_theta, config=t5_config)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, input_ids) -> torch.Tensor:
        outputs = self.hf_module(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


def get_te_model_and_inputs(
    hf_model_name, component, weight_file, batch_size, max_length
):
    match component:
        case "clip":
            te = HFEmbedder(
                "clip",
                77,
                weight_file,
            )
            clip_ids_shape = (
                batch_size,
                77,
            )
            input_args = [
                torch.ones(clip_ids_shape, dtype=torch.int64),
            ]
            return te, input_args
        case "t5xxl":
            te = HFEmbedder(
                "t5xxl",
                512,
                weight_file,
            )
            clip_ids_shape = (
                batch_size,
                512,
            )
            input_args = [
                torch.ones(clip_ids_shape, dtype=torch.int64),
            ]
            return te, input_args


class FluxAEWrapper(torch.nn.Module):
    def __init__(self, weight_file, height=1024, width=1024, precision="fp32"):
        super().__init__()
        dtype = torch_dtypes[precision]
        dataset = Dataset.load(weight_file)
        self.ae = VaeDecoderModel.from_dataset(dataset)
        self.height = height
        self.width = width

    def forward(self, z):
        return self.ae.forward(z)


def get_ae_model_and_inputs(
    hf_model_name, weight_file, precision, batch_size, height, width
):
    dtype = torch_dtypes[precision]
    aeparams = fluxconfigs[hf_model_name].ae_params
    aeparams.height = height
    aeparams.width = width
    ae = FluxAEWrapper(weight_file, height, width, precision).to(dtype)
    latents_shape = (
        batch_size,
        int(height * width / 256),
        64,
    )
    img_shape = (
        1,
        aeparams.in_channels,
        int(height),
        int(width),
    )
    encode_inputs = [
        torch.empty(img_shape, dtype=dtype),
    ]
    decode_inputs = [
        torch.empty(latents_shape, dtype=dtype),
    ]
    return ae, encode_inputs, decode_inputs


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps


class FluxScheduler(torch.nn.Module):
    def __init__(self, max_length, torch_dtype, is_schnell=False):
        super().__init__()
        self.is_schnell = is_schnell
        self.max_length = max_length
        timesteps = [torch.empty((100), dtype=torch_dtype, requires_grad=False)] * 100
        for i in range(1, 100):
            schedule = get_schedule(i, max_length, shift=not self.is_schnell)
            timesteps[i] = torch.nn.functional.pad(schedule, (0, 99 - i), "constant", 0)
        self.timesteps = torch.stack(timesteps, dim=0).clone().detach()

    def prepare(self, num_steps):
        timesteps = self.timesteps[num_steps]
        return timesteps


def get_scheduler_model_and_inputs(hf_model_name, max_length, precision):
    is_schnell = "schnell" in hf_model_name
    mod = FluxScheduler(
        max_length=max_length,
        torch_dtype=torch_dtypes[precision],
        is_schnell=is_schnell,
    )
    sample_inputs = (torch.empty(1, dtype=torch.int64),)
    return mod, sample_inputs


@torch.no_grad()
def export_flux_model(
    hf_model_name,
    component,
    batch_size,
    height,
    width,
    precision="fp16",
    max_length=512,
    compile_to="torch",
    weights_directory=None,
    external_weights=None,
    external_weight_path=None,
    decomp_attn=False,
    split_model=False,
):
    weights_path = Path(weights_directory) / f"exported_parameters_{precision}"
    dtype = torch_dtypes[precision]
    decomp_list = []
    if decomp_attn == True:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        if component == "mmdit":
            # Standard non-split model
            if not split_model:
                model, sample_inputs = get_flux_model_and_inputs(
                    weights_path / f"transformer.{external_weights}",
                    precision,
                    batch_size,
                    max_length,
                    height,
                    width,
                )

                fxb = FxProgramsBuilder(model)

                @fxb.export_program(
                    args=(sample_inputs,),
                )
                def _forward(
                    module,
                    inputs,
                ):
                    return module.forward(*inputs)

                class CompiledFluxTransformer(CompiledModule):
                    run_forward = _forward

                inst = CompiledFluxTransformer(context=Context(), import_to="IMPORT")
                module = CompiledModule.get_mlir_module(inst)
                
            # Split model into front and back halves
            else:
                # Get both models and their inputs
                (front_model, back_model), (front_inputs, back_inputs) = get_flux_transformer_model(
                    weights_path / f"transformer.{external_weights}",
                    img_height=height,
                    img_width=width,
                    compression_factor=8,
                    max_len=max_length,
                    precision=precision,
                    bs=batch_size,
                    split_model=True,
                )
                
                # Create front model module
                fxb_front = FxProgramsBuilder(front_model)
                @fxb_front.export_program(
                    args=(front_inputs,),
                )
                def _forward_front(
                    module,
                    inputs,
                ):
                    return module.forward(*inputs)
                
                # Create back model module
                fxb_back = FxProgramsBuilder(back_model)
                @fxb_back.export_program(
                    args=(back_inputs,),
                )
                def _forward_back(
                    module,
                    inputs,
                ):
                    return module.forward(*inputs)
                
                # Compile both modules
                class CompiledFluxFrontTransformer(CompiledModule):
                    run_forward_front = _forward_front
                
                class CompiledFluxBackTransformer(CompiledModule):
                    run_forward_back = _forward_back
                
                # We'll return both modules as a tuple
                inst_front = CompiledFluxFrontTransformer(context=Context(), import_to="IMPORT")
                inst_back = CompiledFluxBackTransformer(context=Context(), import_to="IMPORT")
                
                module_front = CompiledModule.get_mlir_module(inst_front)
                module_back = CompiledModule.get_mlir_module(inst_back)
                
                # Combine both modules into a single MLIR module
                # For now, just return the front module, we'll handle returning both separately
                module = module_front

        elif component == "clip":
            model, sample_inputs = get_te_model_and_inputs(
                hf_model_name,
                component,
                weights_path / f"clip.{external_weights}",
                batch_size,
                max_length,
            )

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_inputs,),
            )
            def _forward(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledFluxTextEncoder(CompiledModule):
                encode_prompts = _forward

            inst = CompiledFluxTextEncoder(context=Context(), import_to="IMPORT")
            module = CompiledModule.get_mlir_module(inst)
            
        elif component == "t5xxl":
            model, sample_inputs = get_te_model_and_inputs(
                hf_model_name,
                component,
                weights_path / f"t5.{external_weights}",
                batch_size,
                max_length,
            )

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_inputs,),
            )
            def _forward(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledFluxTextEncoder2(CompiledModule):
                encode_prompts = _forward

            inst = CompiledFluxTextEncoder2(context=Context(), import_to="IMPORT")
            module = CompiledModule.get_mlir_module(inst)
            
        elif component == "vae":
            model, encode_inputs, decode_inputs = get_ae_model_and_inputs(
                hf_model_name,
                weights_path / f"vae.{external_weights}",
                precision,
                batch_size,
                height,
                width,
            )

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(decode_inputs,),
            )
            def _decode(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledFluxAutoEncoder(CompiledModule):
                decode = _decode

            inst = CompiledFluxAutoEncoder(context=Context(), import_to="IMPORT")
            module = CompiledModule.get_mlir_module(inst)

        elif component == "scheduler":
            model, sample_inputs = get_scheduler_model_and_inputs(
                hf_model_name, max_length, precision
            )

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_inputs,),
            )
            def _prepare(
                module,
                inputs,
            ):
                return module.prepare(*inputs)

            class CompiledFlowScheduler(CompiledModule):
                run_prep = _prepare

            inst = CompiledFlowScheduler(context=Context(), import_to="IMPORT")
            module = CompiledModule.get_mlir_module(inst)

    # For split model, return both front and back modules
    if split_model and component == "mmdit":
        return str(module_front), str(module_back)
    else:
        # For all other cases, return single module
        return str(module)


def get_filename(args):
    # Handle split models for mmdit
    if args.component == "mmdit" and args.split_model:
        # Generate names for front and back components
        front_suffix = f"mmdit_front_double_end_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}"
        back_suffix = f"mmdit_back_single_start_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}"
        
        front_name = create_safe_name(args.model, front_suffix)
        back_name = create_safe_name(args.model, back_suffix)
        
        return front_name, back_name
    
    # Handle regular models
    match args.component:
        case "mmdit":
            return create_safe_name(
                args.model,
                f"mmdit_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}",
            )
        case "clip":
            return create_safe_name(
                args.model, f"clip_bs{args.batch_size}_77_{args.precision}"
            )
        case "t5xxl":
            return create_safe_name(
                args.model, f"t5xxl_bs{args.batch_size}_256_{args.precision}"
            )
        case "scheduler":
            return create_safe_name(
                args.model,
                f"scheduler_bs{args.batch_size}_{args.max_length}_{args.precision}",
            )
        case "vae":
            return create_safe_name(
                args.model,
                f"vae_bs{args.batch_size}_{args.height}x{args.width}_{args.precision}",
            )


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig(level=logging.DEBUG)
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="flux-schnell",
        choices=["flux-dev", "flux-schnell", "flux-pro"],
    )
    p.add_argument(
        "--component",
        default="mmdit",
        choices=["mmdit", "clip", "t5xxl", "scheduler", "vae"],
    )
    p.add_argument("--batch_size", default=1)
    p.add_argument("--height", default=1024)
    p.add_argument("--width", default=1024)
    p.add_argument("--precision", default="fp32")
    p.add_argument("--max_length", default=512)
    p.add_argument("--weights_directory", default="/data/flux/flux/FLUX.1-dev/")
    p.add_argument("--external_weights", default="irpa")
    p.add_argument("--external_weights_file", default=None)
    p.add_argument("--decomp_attn", action="store_true")
    
    # Simple flag to control splitting at double/single boundary
    p.add_argument("--split_model", action="store_true", 
                   help="Split the model at the boundary between double and single blocks")
    
    args = p.parse_args()

    if args.external_weights and not args.external_weights_file:
        args.external_weights_file = (
            create_safe_name(
                args.model,
                args.component + "_" + args.precision,
            )
            + "."
            + args.external_weights
        )
    
    # For split model
    if args.split_model and args.component == "mmdit":
        # Set up args for get_filename
        args.split_model = True
        
        front_name, back_name = get_filename(args)
        front_mod_str, back_mod_str = export_flux_model(
            args.model,
            args.component,
            args.batch_size,
            args.height,
            args.width,
            args.precision,
            args.max_length,
            "mlir",
            args.weights_directory,
            args.external_weights,
            args.external_weights_file,
            args.decomp_attn,
            split_model=True,
        )

        # Write front model
        with open(f"{front_name}.mlir", "w+") as f:
            f.write(front_mod_str)
        print("Saved front model to", front_name + ".mlir")
        
        # Write back model
        with open(f"{back_name}.mlir", "w+") as f:
            f.write(back_mod_str)
        print("Saved back model to", back_name + ".mlir")
    else:
        # Standard non-split model
        safe_name = get_filename(args)
        mod_str = export_flux_model(
            args.model,
            args.component,
            args.batch_size,
            args.height,
            args.width,
            args.precision,
            args.max_length,
            "mlir",
            args.weights_directory,
            args.external_weights,
            args.external_weights_file,
            args.decomp_attn,
        )

        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
