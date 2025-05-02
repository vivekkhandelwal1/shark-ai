# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "elementwise_tensor_tensor",
]


@CustomOp.register(library=LIBRARY)
class elementwise_tensor_tensor(CustomOp):
    """TODO"""

    signature = "elementwise_tensor_tensor(Tensor x, Tensor y, str op) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        x_desc = ksel.arg_tensor(0)
        y_desc = ksel.arg_tensor(1)
        op_str = ksel.attr_str(2).v

        # Shape match
        torch._check(
            x_desc.t.shape == y_desc.t.shape,
            lambda: f"elementwise_tensor_tensor: Shapes don't match. {x_desc.t.shape} != {y_desc.t.shape}",
        )
        torch._check(
            x_desc.t.dtype == y_desc.t.dtype,
            lambda: f"elementwise_tensor_tensor: Types don't match. {x_desc.t.dtype} != {y_desc.t.dtype}",
        )

        ksel.return_new_tensor(x_desc.t.shape, dtype=x_desc.t.dtype)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        x = kb.arg_value(0)
        x_tensor_type = RankedTensorType(x.type)
        op_str = ksel.arg_descs[2].v

        rank = x_tensor_type.rank
        element_type = x_tensor_type.element_type

        template_file = "elementwise_tensor_tensor.mlir"
        target_function_name = (
            f"sharktank_elementwise_tensor_tensor_{op_str}_{rank}_{element_type}"
        )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            op_str=op_str,
            rank=rank,
            dtype=element_type,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
