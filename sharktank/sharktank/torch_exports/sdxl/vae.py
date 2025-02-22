# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np
from iree.turbine.aot import *

from diffusers.models import AutoencoderKL

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


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

    def decode(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        x = self.vae.decode(latents, return_dict=False)[0]
        return (x / 2 + 0.5).clamp(0, 1)

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return self.vae.config.scaling_factor * latents


@torch.no_grad()
def get_vae_model_and_inputs(
    hf_model_name,
    height,
    width,
    external_weight_path,
    quant_paths,
    num_channels=4,
    precision="fp16",
    batch_size=1,
):
    dtype = torch_dtypes[precision]
    vae_model = get_vae_model(
        hf_model_name,
        external_weight_path,
        quant_paths,
        precision,
    )

    input_image_shape = (batch_size, 3, height, width)
    input_latents_shape = (batch_size, num_channels, height // 8, width // 8)
    # encode_args = [
    #    torch.rand(
    #        input_image_shape,
    #        dtype=dtype,
    #    )
    # ]
    decode_args = [
        torch.rand(
            input_latents_shape,
            dtype=dtype,
        ),
    ]
    return vae_model, decode_args


def get_vae_model(hf_model_name, external_weight_path, quant_paths, quantize=True):
    from sharktank.models.vae.model import VaeDecoderModel

    if quantize:
        repo_id = "amd-shark/sdxl-quant-fp8"
        subfolder = "unet_int8_sdpa_fp8_vae_int8"
        revision = "f61f04ffc19a38bb56ebd045510e7e1b031d56fe"

    def download(filename):
        return hf_hub_download(
            repo_id=repo_id, subfolder=subfolder, filename=filename, revision=revision
        )

    # if quant_paths and quant_paths["config"] and os.path.exists(quant_paths["config"]):
    #    results = {
    #        "config.json": quant_paths["config"],
    #    }
    # else:
    #    results = {
    #        "config.json": download("config.json")
    #    }
    results = {
        "config.json": hf_hub_download(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="vae",
            filename="config.json",
        )
    }
    if quant_paths and quant_paths["params"] and os.path.exists(quant_paths["params"]):
        results["vae_params.safetensors"] = quant_paths["params"]
    else:
        results["vae_params.safetensors"] = download("vae_params.safetensors")

    output_dir = os.path.dirname(external_weight_path)

    if (
        quant_paths
        and quant_paths["vae_quant_params"]
        and os.path.exists(quant_paths["vae_quant_params"])
    ):
        results["vae_quant_params.json"] = quant_paths["vae_quant_params"]
    else:
        results["vae_quant_params.json"] = download("vae_quant_params.json")
    ds_filename = os.path.basename(external_weight_path)
    output_path = os.path.join(output_dir, ds_filename)
    ds = get_vae_dataset(
        results["config.json"],
        results["vae_params.safetensors"],
        output_path,
        results["vae_quant_params.json"],
    )

    vae = VaeDecoderModel.from_dataset(ds)
    ds.save(output_path)
    return vae


def get_vae_dataset(
    config_json_path,
    params_path,
    output_path,
    quant_params_path=None,
):
    from sharktank.models.punet.tools import import_brevitas_dataset

    ds_import_args = [
        f"--config-json={config_json_path}",
        f"--params={params_path}",
        f"--output-irpa-file={output_path}",
    ]
    if quant_params_path:
        ds_import_args.extend([f"--quant-params={quant_params_path}"])
    import_brevitas_dataset.main(ds_import_args)
    return import_brevitas_dataset.Dataset.load(output_path)
