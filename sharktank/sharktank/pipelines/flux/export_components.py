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


class FluxSectionDenoiseStepModel(torch.nn.Module):
    """Denoising step model that processes a specific section of the model.
    
    This allows splitting the model into front and back components for memory-efficient
    processing, where each component handles a specific section of the computation.
    """
    def __init__(
        self,
        theta,
        params,
        batch_size=1,
        max_length=512,
        height=1024,
        width=1024,
        start_point="input",  # "input", "double_N", "single_0"
        end_point="output",   # "double_N", "single_N", "output"
    ):
        super().__init__()
        self.params = params
        self.batch_size = batch_size
        self.start_point = start_point
        self.end_point = end_point
        
        # Set up position embeddings
        img_ids = torch.zeros(height // 16, width // 16, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 16)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 16)[None, :]
        self.img_ids = img_ids.reshape(1, height * width // 256, 3)
        self.txt_ids = torch.zeros(1, max_length, 3)
        
        # Determine which components are needed based on start/end points
        self._setup_model(theta, params)
        
    def _parse_block_index(self, point_name):
        """Parse a block spec like 'double_3' or 'single_2' into type and index."""
        if point_name == "input":
            return "input", 0
        elif point_name == "output":
            return "output", 0
            
        parts = point_name.split("_")
        if len(parts) != 2 or parts[0] not in ["double", "single"]:
            raise ValueError(f"Invalid block specification: {point_name}")
            
        try:
            index = int(parts[1])
            return parts[0], index
        except ValueError:
            raise ValueError(f"Invalid block index in: {point_name}")
            
    def _validate_points(self, start, end, params):
        """Validate that start and end points form a valid range."""
        # Convert logical points to numeric values for comparison
        start_type, start_idx = start
        end_type, end_idx = end
        
        # Map types to numeric values for ordering (input < double < single < output)
        type_order = {"input": 0, "double": 1, "single": 2, "output": 3}
        
        # Check type order
        if type_order[start_type] > type_order[end_type]:
            raise ValueError(f"Start point {self.start_point} must come before end point {self.end_point}")
            
        # If same type, check indices
        if start_type == end_type and start_type in ["double", "single"]:
            if start_idx > end_idx:
                raise ValueError(f"Start index {start_idx} must be <= end index {end_idx}")
                
        # Check block indices against model params
        if start_type == "double" and start_idx >= params.depth:
            raise ValueError(f"Double block index {start_idx} exceeds model depth {params.depth}")
        if end_type == "double" and end_idx >= params.depth:
            raise ValueError(f"Double block index {end_idx} exceeds model depth {params.depth}")
        if start_type == "single" and start_idx >= params.depth_single_blocks:
            raise ValueError(f"Single block index {start_idx} exceeds model single depth {params.depth_single_blocks}")
        if end_type == "single" and end_idx >= params.depth_single_blocks:
            raise ValueError(f"Single block index {end_idx} exceeds model single depth {params.depth_single_blocks}")
            
    def _setup_model(self, theta, params):
        """Set up model components based on the specified start and end points."""
        start = self._parse_block_index(self.start_point)
        end = self._parse_block_index(self.end_point)
        
        # Validate points make sense
        self._validate_points(start, end, params)
        
        # Create position embedder (needed for all sections)
        self.pe_embedder = EmbedND(
            dim=params.hidden_size // params.num_heads, 
            theta=params.theta, 
            axes_dim=params.axes_dim
        )
        
        # Input processing components if starting from input
        if start[0] == "input":
            self.img_in = LinearLayer(theta("img_in"))
            self.time_in = MLPEmbedder(theta("time_in"))
            self.vector_in = MLPEmbedder(theta("vector_in"))
            self.guidance = params.guidance_embed
            if params.guidance_embed:
                self.guidance_in = MLPEmbedder(theta("guidance_in"))
            self.txt_in = LinearLayer(theta("txt_in"))
        
        # Double blocks
        if (start[0] in ["input", "double"] and end[0] in ["double", "single", "output"]):
            start_idx = start[1] if start[0] == "double" else 0
            end_idx = end[1] if end[0] == "double" else params.depth
            
            if start_idx < end_idx:
                self.double_blocks = torch.nn.ModuleList(
                    [
                        MMDITDoubleBlock(
                            theta("double_blocks", i),
                            num_heads=params.num_heads,
                            hidden_size=params.hidden_size,
                        )
                        for i in range(start_idx, end_idx)
                    ]
                )
            else:
                self.double_blocks = torch.nn.ModuleList([])
        
        # Single blocks
        if (start[0] in ["input", "double", "single"] and end[0] in ["single", "output"]):
            # Include single blocks only if we're past all double blocks or starting at singles
            if start[0] == "single" or (start[0] in ["input", "double"] and 
                                       (end[0] != "double" or start[1] >= params.depth)):
                start_idx = start[1] if start[0] == "single" else 0
                end_idx = end[1] + 1 if end[0] == "single" else params.depth_single_blocks
                
                self.single_blocks = torch.nn.ModuleList(
                    [
                        MMDITSingleBlock(
                            theta("single_blocks", i),
                            num_heads=params.num_heads,
                            hidden_size=params.hidden_size,
                        )
                        for i in range(start_idx, end_idx)
                    ]
                )
            else:
                self.single_blocks = None
        
        # Final layer
        if end[0] == "output":
            self.final_layer = LastLayer(theta("final_layer"))

    def forward(self, img, txt, vec, step, timesteps, guidance_scale, 
                img_inter=None, txt_inter=None, t_vec=None, t_curr=None, 
                combined_inter=None):
        """Process the specified section of the model.
        
        Args:
            img: Input image latents (required when starting from input)
            txt: Input text embeddings (required when starting from input)
            vec: Input CLIP features (required when starting from input)
            step: Current denoising step (required when starting from input)
            timesteps: Timestep schedule (required)
            guidance_scale: CFG guidance scale (required when starting from input)
            
            img_inter: Intermediate image state (required when not starting from input)
            txt_inter: Intermediate text state (required when not starting from input)
            t_vec: Processed timestep vector (required when not starting from input)
            t_curr: Current timestep (required when starting from input or ending at output)
            combined_inter: Combined img+txt representation for single blocks (required for starting at single)
            
        Returns:
            Appropriate intermediate state or final image based on end_point
        """
        # Starting from input - do initial processing
        if self.start_point == "input":
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
            
        # Starting from intermediate state
        else:
            if self.start_point.startswith("double"):
                if img_inter is None or txt_inter is None or t_vec is None:
                    raise ValueError(f"Starting from {self.start_point} requires img_inter, txt_inter, and t_vec")
                    
                img_processed = img_inter
                txt_processed = txt_inter
                vec_processed = t_vec
                
            elif self.start_point.startswith("single"):
                if combined_inter is None or t_vec is None:
                    raise ValueError(f"Starting from {self.start_point} requires combined_inter and t_vec")
                    
                combined = combined_inter
                vec_processed = t_vec
        
        # Process through blocks
        ids = torch.cat((self.txt_ids, self.img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        # Process double blocks if present
        if hasattr(self, 'double_blocks') and len(self.double_blocks) > 0:
            for block in self.double_blocks:
                img_processed, txt_processed = block(
                    img=img_processed, txt=txt_processed, vec=vec_processed, pe=pe
                )
                
        # Process single blocks if present and we've moved past double blocks
        if hasattr(self, 'single_blocks') and self.single_blocks is not None:
            # Combine image and text for single blocks if coming from double blocks
            if not hasattr(self, 'combined'):
                combined = torch.cat((txt_processed, img_processed), 1)
                
            # Process through single blocks
            for block in self.single_blocks:
                combined = block(combined, vec=vec_processed, pe=pe)
                
            # Extract image part if we're going to final layer
            if self.end_point == "output":
                img_processed = combined[:, txt_processed.shape[1]:, ...]
        
        # Apply final layer and timestep scaling if ending at output
        if self.end_point == "output":
            # Final projection
            pred = self.final_layer(img_processed, vec_processed)
            
            # Apply timestep scaling
            if t_curr is None:
                raise ValueError("t_curr is required when end_point is 'output'")
                
            t_prev = torch.index_select(timesteps, 0, step + 1)
            if self.start_point == "input":
                updated_img = img + (t_prev - t_curr) * pred
            else:
                updated_img = img_inter + (t_prev - t_curr) * pred
                
            return updated_img
        
        # Return appropriate intermediate state based on where we stopped
        elif self.end_point.startswith("double"):
            return img_processed, txt_processed, vec_processed, t_curr
            
        elif self.end_point.startswith("single"):
            return combined, vec_processed, t_curr


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
    front_endpoint="double_0",  # If split_model is True, endpoint for front half
    back_startpoint="double_0",  # If split_model is True, startpoint for back half
):
    # DNS: refactor file to select datatype
    dtype = torch_dtypes[precision]
    transformer_dataset = Dataset.load(weight_file)
    flux_params = FluxParams.from_hugging_face_properties(transformer_dataset.properties)
    
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
        # Create front half model
        front_model = FluxSectionDenoiseStepModel(
            theta=transformer_dataset.root_theta,
            params=flux_params,
            start_point="input",
            end_point=front_endpoint,
        )
        
        # Create back half model
        back_model = FluxSectionDenoiseStepModel(
            theta=transformer_dataset.root_theta,
            params=flux_params,
            start_point=back_startpoint,
            end_point="output",
        )
        
        # Prepare sample inputs
        sample_args, sample_kwargs = front_model.mmdit.sample_inputs() if hasattr(front_model, 'mmdit') else FluxModelV1(theta=transformer_dataset.root_theta, params=flux_params).sample_inputs()
        front_inputs = (
            sample_kwargs["img"],
            sample_kwargs["txt"],
            sample_kwargs["y"],
            torch.full((bs,), 1, dtype=torch.int64),
            torch.full((100,), 1, dtype=dtype),
            sample_kwargs["guidance"],
        )
        
        # For back half, we need different inputs based on the split point
        if back_startpoint.startswith("double"):
            # Back half expects (img_inter, txt_inter, t_vec, t_curr, img_orig, step, timesteps)
            back_inputs = (
                sample_kwargs["img"],  # placeholder for img_inter
                sample_kwargs["txt"],  # placeholder for txt_inter
                torch.full((bs,), 1, dtype=dtype),  # placeholder for t_vec
                torch.full((bs,), 1, dtype=dtype),  # placeholder for t_curr
                sample_kwargs["img"],  # placeholder for original img
                torch.full((bs,), 1, dtype=torch.int64),  # step
                torch.full((100,), 1, dtype=dtype),  # timesteps
            )
        elif back_startpoint.startswith("single"):
            # Back half expects (combined_inter, t_vec, t_curr, [other params])
            combined_shape = list(sample_kwargs["txt"].shape)
            combined_shape[1] += sample_kwargs["img"].shape[1]
            back_inputs = (
                torch.zeros(combined_shape, dtype=dtype),  # placeholder for combined_inter
                torch.full((bs,), 1, dtype=dtype),  # placeholder for t_vec
                torch.full((bs,), 1, dtype=dtype),  # placeholder for t_curr
                sample_kwargs["img"],  # placeholder for original img
                torch.full((bs,), 1, dtype=torch.int64),  # step
                torch.full((100,), 1, dtype=dtype),  # timesteps
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
    front_endpoint="double_0",
    back_startpoint="double_0",
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
                    front_endpoint=front_endpoint,
                    back_startpoint=back_startpoint,
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
                inst_front = CompiledFluxFrontTransformer(context=Context(), import_to="IMPORT_FRONT")
                inst_back = CompiledFluxBackTransformer(context=Context(), import_to="IMPORT_BACK")
                
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
        front_suffix = f"mmdit_front_{args.front_endpoint}_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}"
        back_suffix = f"mmdit_back_{args.back_startpoint}_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}"
        
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
    
    # Add a single parameter to control splitting
    p.add_argument("--split_point", default=None, 
                   help="Split the model at this point (e.g., 'double_3'). Creates front and back halves.")
    
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
    
    # Determine if we're creating a split model
    split_model = args.split_point is not None and args.component == "mmdit"
    
    # For split model, use the specified split point
    if split_model:
        # Set up args for get_filename
        args.split_model = True
        args.front_endpoint = args.split_point
        args.back_startpoint = args.split_point
        
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
            front_endpoint=args.split_point,
            back_startpoint=args.split_point,
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
