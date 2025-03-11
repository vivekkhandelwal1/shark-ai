# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from os import PathLike
import os
from pathlib import Path
import torch

from ...export import export_static_model_mlir
from ...tools.import_hf_dataset import import_hf_dataset
from .flux import FluxModelV1, FluxParams
from ...types import Dataset
from ...types.tensors import PrimitiveTensor, DefaultPrimitiveTensor                                                                                                                                                                                                                                                                                          
from ...types.theta import Theta
from ...utils.hf_datasets import get_dataset
from sharktank.transforms.dataset import set_float_dtype

flux_transformer_default_batch_sizes = [1]


def combine_mmdit_data(theta: Theta) -> Theta:
    """Transforms the theta by combining img/txt weights in MMDITDoubleBlock."""
    flat_tensors = theta.flatten()
    new_tensors = {}
    
    for name, tensor in flat_tensors.items():
        # Look for pairs to merge
        if isinstance(tensor, PrimitiveTensor) and ("img_attn.qkv" in name or "img_attn.norm" in name):
            # Skip these, we'll be combining these with txt
            continue
        elif isinstance(tensor, PrimitiveTensor) and ("txt_attn.qkv" in name or "txt_attn.norm" in name):
            # Find the corresponding tensor
            img_name = name.replace("txt", "img")

            # Create combined tensor
            img_tensor = flat_tensors[img_name]
            txt_tensor = tensor
            img_data = img_tensor.as_torch()
            txt_data = txt_tensor.as_torch()
            print(name, img_data.shape, txt_data.shape)
            combined_data = torch.cat([img_data, txt_data], dim=0)
            
            # Use the existing img tensor name to ensure the model can find it
            combined_name = name.replace("txt", "combined")
            combined_tensor = DefaultPrimitiveTensor(
                name=combined_name, 
                data=combined_data
            )
            new_tensors[combined_name] = combined_tensor
        else:
            # For all other tensors, keep them as they are
            new_tensors[name] = tensor
    
    return Theta(new_tensors)

def export_flux_transformer_model_mlir(
    model_or_parameters_path: FluxModelV1 | PathLike,
    /,
    output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    if isinstance(model_or_parameters_path, (PathLike, str)):
        dataset = Dataset.load(model_or_parameters_path)
        model = FluxModelV1(
            theta=dataset.root_theta,
            params=FluxParams.from_hugging_face_properties(dataset.properties),
        )
    else:
        model = model_or_parameters_path

    export_static_model_mlir(model, output_path=output_path, batch_sizes=batch_sizes)


def export_flux_transformer_iree_parameters(
    model: FluxModelV1, parameters_output_path: PathLike, dtype=None
):
    model.theta.rename_tensors_to_paths()
    dataset = Dataset(
        root_theta=model.theta, properties=model.params.to_hugging_face_properties()
    )
    if dtype:
        dataset.root_theta = dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=dtype)
        )
    dataset.save(parameters_output_path)


def export_flux_transformer(
    model: FluxModelV1,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    export_flux_transformer_iree_parameters(model, parameters_output_path)
    export_flux_transformer_model_mlir(
        parameters_output_path, output_path=mlir_output_path, batch_sizes=batch_sizes
    )


def import_flux_transformer_dataset_from_hugging_face(
    repo_id: str, parameters_output_path: PathLike
):
    hf_dataset = get_dataset(
        repo_id,
    ).download()

    import_hf_dataset(
        config_json_path=hf_dataset["config"][0],
        param_paths=hf_dataset["parameters"],
        output_irpa_file=parameters_output_path,
    )


def export_flux_transformer_from_hugging_face(
    repo_id: str,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    import_flux_transformer_dataset_from_hugging_face(
        repo_id=repo_id, parameters_output_path=parameters_output_path
    )
    export_flux_transformer_model_mlir(
        parameters_output_path, output_path=mlir_output_path, batch_sizes=batch_sizes
    )


def export_flux_transformer_models(dir: Path):
    from .testing import export_dev_random_single_layer

    base_dir = dir / "flux" / "transformer"
    os.makedirs(base_dir)

    file_name_base = "black-forest-labs--FLUX.1-dev--black-forest-labs-transformer-bf16"
    mlir_path = base_dir / f"{file_name_base}.mlir"
    parameters_output_path = base_dir / f"{file_name_base}.irpa"
    export_flux_transformer_from_hugging_face(
        "black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
        mlir_output_path=mlir_path,
        parameters_output_path=parameters_output_path,
    )

    file_name_base = (
        "black-forest-labs--FLUX.1-schnell--black-forest-labs-transformer-bf16"
    )
    mlir_path = base_dir / f"{file_name_base}.mlir"
    parameters_output_path = base_dir / f"{file_name_base}.irpa"
    export_flux_transformer_from_hugging_face(
        "black-forest-labs/FLUX.1-schnell/black-forest-labs-transformer",
        mlir_output_path=mlir_path,
        parameters_output_path=parameters_output_path,
    )

    file_name_base = "black-forest-labs--FLUX.1-dev--transformer-single-layer-b16"
    mlir_path = base_dir / f"{file_name_base}.mlir"
    parameters_output_path = base_dir / f"{file_name_base}.irpa"
    export_dev_random_single_layer(
        dtype=torch.bfloat16,
        mlir_output_path=mlir_path,
        parameters_output_path=parameters_output_path,
    )
