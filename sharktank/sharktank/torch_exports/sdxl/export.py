# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

from PIL import Image

from collections.abc import Iterable

from iree.compiler.ir import Context
from iree.turbine.aot import *
from iree.turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
from iree.turbine.ops.iree import trace_tensor
import torch
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from diffusers.models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
from ip_adapter.resampler import Resampler
from ip_adapter.attention_processor import IPAttnProcessor2_0, AttnProcessor2_0
from safetensors import safe_open

from huggingface_hub import hf_hub_download

import numpy as np



torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def flatten(xs):
    print("FLATTEN")
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes, torch.Tensor)):
            yield from flatten(x)
        else:
            yield x

class ControlledUnetModel(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel,  controlnet: ControlNetModel, model_ckpt: str, num_tokens, scale):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.do_classifier_free_guidance = True
        self.cross_attention_kwargs = None

    def set_ip_adapter(self, model_ckpt, num_tokens, scale):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0().to(unet.device, dtype=unet.dtype)
            else:
                attn_procs[name] = IPAttnProcessor2_0(hidden_size=hidden_size, 
                                                   cross_attention_dim=cross_attention_dim, 
                                                   scale=scale,
                                                   num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
        unet.set_attn_processor(attn_procs)
        
        state_dict = torch.load(model_ckpt, map_location="cpu")
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        if 'ip_adapter' in state_dict:
            state_dict = state_dict['ip_adapter']
        ip_layers.load_state_dict(state_dict)
    
    def set_ip_adapter_scale(self, scale):
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor2_0):
                attn_processor.scale = scale

    def forward(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor,
        prompt_embeds: torch.Tensor,
        add_text_embeds: torch.Tensor, 
        add_time_ids: torch.Tensor,
        prompt_image_emb: torch.Tensor,
        cond_scale: torch.Tensor,
        guidance_scale: torch.Tensor,
    ):
        encoder_hidden_states = torch.cat([prompt_embeds, prompt_image_emb], dim=1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # controlnet(s) inference
        control_model_input = latents
        controlnet_prompt_embeds = prompt_image_emb
        controlnet_added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }
        control_image = image

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=control_image,
            conditioning_scale=cond_scale,
            guess_mode=False,
            added_cond_kwargs=controlnet_added_cond_kwargs,
            return_dict=False,
        )
        # predict the noise residual
        noise_pred = self.unet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=None,
            cross_attention_kwargs=self.cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

class ResamplerModel(torch.nn.Module):
    def __init__(self, model_ckpt, image_emb_dim=512, num_tokens=16):
        super().__init__()
        self.resampler = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=2048, # unet cross attn dim
            ff_mult=4,
        )
        state_dict = torch.load(model_ckpt, map_location="cpu")
        if 'image_proj' in state_dict:
            state_dict = state_dict["image_proj"]
        self.resampler.load_state_dict(state_dict)
        self.image_proj_model_in_features = image_emb_dim
        self.do_cfg = True


    def forward(
        self,
        prompt_image_emb: torch.Tensor,
    ):
        prompt_image_emb = prompt_image_emb.reshape([1, -1, self.image_proj_model_in_features])
        if self.do_cfg:
            prompt_image_emb = torch.cat([torch.zeros_like(prompt_image_emb), prompt_image_emb], dim=0)
        else:
            prompt_image_emb = torch.cat([prompt_image_emb], dim=0)
        return self.resampler(prompt_image_emb)

class ControlnetModel(torch.nn.Module):
    def __init__(self, controlnet: ControlNetModel):
        super().__init__()
        self.controlnet = controlnet
        self.do_classifier_free_guidance = True
        self.cross_attention_kwargs = None

    def forward(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        image: torch.Tensor,
        prompt_embeds: torch.Tensor,
        add_text_embeds: torch.Tensor, 
        add_time_ids: torch.Tensor,
        prompt_image_emb: torch.Tensor,
        cond_scale: torch.Tensor,
        guidance_scale: torch.Tensor,
    ):
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # controlnet(s) inference
        control_model_input = latents
        controlnet_added_cond_kwargs = added_cond_kwargs

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=prompt_image_emb,
            controlnet_cond=image,
            conditioning_scale=cond_scale,
            guess_mode=False,
            added_cond_kwargs=controlnet_added_cond_kwargs,
            return_dict=False,
        )
        return down_block_res_samples, mid_block_res_sample

class VaeModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        custom_vae="",
    ):
        super().__init__()
        self.vae = None
        if custom_vae in ["", None]:
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
        elif not isinstance(custom_vae, dict):
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
            fp16_weights = hf_hub_download(
                repo_id=custom_vae,
                filename="vae/vae.safetensors",
            )
            with safe_open(fp16_weights, framework="pt", device="cpu") as f:
                state_dict = {}
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
                self.vae.load_state_dict(state_dict)
        else:
            # custom vae as a HF state dict
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
            self.vae.load_state_dict(custom_vae)

    def decode(self, inp):
        inp = 1 / self.vae.config.scaling_factor * inp
        x = self.vae.decode(inp, return_dict=False)[0]
        return (x / 2 + 0.5).clamp(0, 1)

    def encode(self, inp):
        latents = self.vae.encode(inp).latent_dist.sample()
        return self.vae.config.scaling_factor * latents

def get_controlled_model_and_inputs(
    hf_model_name,
    external_weight_path,
    height,
    width,
    controlnet_id = "InstantID/ControlNetModel",
    ip_adapter_id = "InstantID/ip-adapter.bin",
    ip_adapter_scale = 0.5,
    quant_paths = None,
    precision = "fp16",
    batch_size = 1,
    max_length = 64,
    cfg_mode = True,
):
    #unet_model = UNet2DConditionModel.from_pretrained(hf_model_name, subfolder="unet", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "/home/eagarvey/face_swap/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.load_ip_adapter_instantid('InstantID/ip-adapter.bin')
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    pipe.unet.eval()
    pipe.controlnet.eval()

    controlled_unet = ControlledUnetModel(pipe.unet, pipe.controlnet, ip_adapter_id, 16, ip_adapter_scale)

    cfg_dim = 2 if cfg_mode else 1
    dtype = torch_dtypes[precision]
    sample_inputs = {
        "latents": torch.rand([batch_size * cfg_dim, 4, height//8, width//8], dtype=dtype),
        "t": torch.tensor([951.], dtype=dtype),
        "image": torch.rand([batch_size * cfg_dim, 3, height, width], dtype=dtype),
        "prompt_embeds": torch.rand([batch_size * cfg_dim, max_length, 2048], dtype=dtype),
        "add_text_embeds": torch.rand([batch_size * cfg_dim, 1280], dtype=dtype),
        "add_time_ids": torch.zeros([batch_size * cfg_dim, 6], dtype=dtype),
        "prompt_image_emb": torch.rand([batch_size * cfg_dim, 16, 2048], dtype=dtype),
        "cond_scale": torch.tensor([0.8], dtype=dtype),
        "guidance_scale": torch.tensor([7.5], dtype=dtype),
    }
    return controlled_unet, list(sample_inputs.values())

def get_controlnet_model_and_inputs(
    hf_model_name,
    external_weight_path,
    height,
    width,
    controlnet_id = "InstantID/ControlNetModel",
    ip_adapter_id = "InstantID/ip-adapter.bin",
    ip_adapter_scale = 0.5,
    quant_paths = None,
    precision = "fp16",
    batch_size = 1,
    max_length = 64,
    cfg_mode = True,
):
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
    control_mod = ControlnetModel(controlnet)
    cfg_dim = 2 if cfg_mode else 1
    dtype = torch_dtypes[precision]
    sample_inputs = {
        "latents": torch.rand([batch_size, 4, height//8, width//8], dtype=dtype),
        "t": torch.zeros(1, dtype=dtype),
        "image": torch.rand([batch_size * cfg_dim, 3, height, width], dtype=dtype),
        "prompt_embeds": torch.rand([batch_size * cfg_dim, max_length, 2048], dtype=dtype),
        "add_text_embeds": torch.rand([batch_size * cfg_dim, 1280], dtype=dtype),
        "add_time_ids": torch.tensor([height, width, 0, 0, height, width], dtype=dtype),
        "prompt_image_emb": torch.rand([batch_size * cfg_dim, 16, 2048], dtype=dtype),
        "cond_scale": torch.tensor([0.8], dtype=dtype),
        "guidance_scale": torch.tensor([7.5], dtype=dtype),
    }
    return control_mod, list(sample_inputs.values())

def get_vae_model_and_inputs(
    hf_model_name,
    height,
    width,
    num_channels = 4,
    precision = "fp16",
    batch_size = 1,
):
    dtype = torch_dtypes[precision]
    if dtype == torch.float16:
        custom_vae = "amd-shark/sdxl-quant-models"
    else:
        custom_vae = None
    vae_model = VaeModel(hf_model_name, custom_vae=custom_vae).to(dtype=dtype)
    input_image_shape = (1, 3, height, width)
    input_latents_shape = (batch_size, num_channels, height // 8, width // 8)
    encode_args = [
        torch.rand(
            input_image_shape,
            dtype=dtype,
        )
    ]
    decode_args = [
        torch.empty(
            input_latents_shape,
            dtype=dtype,
        )
    ]
    return vae_model, encode_args, decode_args
    
def create_safe_name(hf_model_name, model_name_str=""):
    if not model_name_str:
        model_name_str = ""
    if model_name_str != "" and (not model_name_str.startswith("_")):
        model_name_str = "_" + model_name_str

    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub("\.", "_", safe_name)
    return safe_name

@torch.no_grad()
def export_sdxl_model(
    hf_model_name,
    component,
    batch_size,
    height,
    width,
    precision="fp16",
    max_length=64,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    decomp_attn=False,
    save_goldens=True,
):
    dtype = torch_dtypes[precision]
    decomp_list = [torch.ops.aten.logspace]
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
        if component == "controlled_unet":
            model, sample_inputs = get_controlled_model_and_inputs(
                hf_model_name, external_weight_path, height, width
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

            class CompiledControlledUnet(CompiledModule):
                run_forward = _forward

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledControlledUnet(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)
            sample_input_list = [sample_inputs]
            golden_outputs = [model.forward(*sample_inputs)]
        elif component == "vae":
            model, encode_args, decode_args = get_vae_model_and_inputs(hf_model_name, height, width, precision=precision, batch_size=batch_size)
            fxb = FxProgramsBuilder(model)

            # # TODO: fix issues with exporting the encode function.
            # @fxb.export_program(args=(encode_args,))
            # def _encode(module, inputs,):
            #     return module.encode(*inputs)

            @fxb.export_program(args=(decode_args,))
            def _decode(module, inputs):
                return module.decode(*inputs)

            class CompiledVae(CompiledModule):
                decode = _decode
                # encode = _encode

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledVae(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)
            sample_input_list = [encode_args, decode_args]
            golden_out_encode = model.encode(*encode_args)
            golden_out_decode = model.decode(*decode_args)
            golden_outputs = [golden_out_encode, golden_out_decode]

        elif component == "controlnet":
            model, sample_inputs = get_controlnet_model_and_inputs(
                hf_model_name, external_weight_path, height, width
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

            class CompiledControlnet(CompiledModule):
                run_forward = _forward

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledControlnet(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)
            sample_input_list = [sample_inputs]
            out = model.forward(*sample_inputs)
            golden_outputs = [flatten(out)]


        elif component == "resampler":
            ckpt_path = 'InstantID/ip-adapter.bin'
            model = ResamplerModel(ckpt_path).to(dtype)
            sample_inputs = [torch.rand([512], dtype=dtype)]
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_inputs,),
            )
            def _forward(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledResampler(CompiledModule):
                run_image_proj = _forward

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledResampler(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)
            sample_input_list = [sample_inputs]
            golden_outputs = [model.forward(*sample_inputs)]

        elif component == "scheduler":
            from scheduler import SchedulingModel, get_scheduler, get_sample_sched_inputs
            scheduler_id = "EulerDiscrete"
            scheduler = get_scheduler(hf_model_name, scheduler_id)
            scheduler_module = SchedulingModel(
                hf_model_name, scheduler, height, width, batch_size, dtype
            )
            scheduler_module.do_classifier_free_guidance = True
            init_args, prep_args, step_args = get_sample_sched_inputs(batch_size, height, width, dtype)
            fxb = FxProgramsBuilder(scheduler_module)

            @fxb.export_program(
                args=(init_args,),
            )
            def _initialize(module, sample):
                return module.initialize(*sample)

            @fxb.export_program(
                args=(prep_args,),
            )
            def _scale(module, inputs):
                return module.prepare_model_input(*inputs)

            @fxb.export_program(
                args=(step_args,),
            )
            def _step(module, inputs):
                return module.step(*inputs)

            class CompiledScheduler(CompiledModule):
                run_initialize = _initialize
                run_scale = _scale
                run_step = _step

            import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
            inst = CompiledScheduler(context=Context(), import_to=import_to)

            module = CompiledModule.get_mlir_module(inst)
            sample_input_list = [init_args, prep_args, step_args]
            golden_outputs = [scheduler_module.step(*step_args)]
        else:
            raise ValueError("Unimplemented: ", component)
    if save_goldens:
        for idx, i in enumerate(sample_input_list):
            for num, inp in enumerate(i):
                np.save(f"{component}_{idx}_input_{num}.npy", inp)
        for idx, i in enumerate(golden_outputs):
            for num, out in enumerate(i):
                np.save(f"{component}_{num}_golden_out_{idx}.npy", np.asarray(out))
    module_str = str(module)
    return module_str


def get_filename(args):
    match args.component:
        case "controlled_unet":
            return create_safe_name(
                args.model,
                f"controlled_unet_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}",
            )
        case "controlnet":
            return create_safe_name(
                args.model,
                f"controlnet_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}",
            )
        case "unet":
            return create_safe_name(
                args.model,
                f"unet_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}",
            )
        case "clip":
            return create_safe_name(
                args.model, f"clip_bs{args.batch_size}_{args.max_length}_{args.precision}"
            )
        case "vae":
            return create_safe_name(
                args.model,
                f"vae_bs{args.batch_size}_{args.height}x{args.width}_{args.precision}",
            )
        case "resampler":
            return create_safe_name(
                args.model,
                f"resampler_bs{args.batch_size}_512_{args.precision}",
            )
        case "scheduler":
            return create_safe_name(
                args.model,
                f"scheduler_bs{args.batch_size}_{args.height}x{args.width}_{args.precision}"
            )


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig(level=logging.DEBUG)
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    p.add_argument(
        "--component",
        default="controlled_unet",
        choices=["unet", "controlled_unet", "controlnet", "clip", "scheduler", "vae", "resampler", "scheduler"],
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--precision", default="fp16")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--external_weights", default="irpa")
    p.add_argument("--external_weights_path", default=None)
    p.add_argument("--decomp_attn", action="store_true")
    args = p.parse_args()

    if args.external_weights and not args.external_weights_path:
        args.external_weights_path = (
            create_safe_name(
                args.model,
                args.component + "_" + args.precision,
            )
            + "."
            + args.external_weights
        )
    safe_name = get_filename(args)
    mod_str = export_sdxl_model(
        args.model,
        args.component,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.max_length,
        "mlir",
        args.external_weights,
        args.external_weights_path,
        args.decomp_attn,
    )

    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
