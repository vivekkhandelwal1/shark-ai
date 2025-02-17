# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "mmt_block_scaled_f8",
]


@CustomOp.register(library=LIBRARY)
class mmt_block_scaled_f8(CustomOp):
    """
    Generic block-quantized fp8E5M2FNUZ matmul with transposed RHS, 
    corresponding to the following dequantized mmt:

    DQuant_LHS : [M, K] f32
    DQuant_RHS : [N, K] f32

    The inputs to this op should follow the GenericBlockScaledLayout. 
    That is, they should be expanded by blocks as:

    LHS : [num_blocks_M, block_size_M, num_blocks_K, block_size_K]
    RHS : [num_blocks_N, block_size_N, num_blocks_K, block_size_K]

    and the scale tensors should be of the form:

    LHS_Scale : [num_blocks_M, 1, num_blocks_K, 1]
    RHS_Scale : [num_blocks_N, 1, num_blocks_K, 1]
    """

    signature = "mmt_block_scaled_f8(Tensor a, Tensor b, Tensor a_s, Tensor b_s) -> (Tensor)"

    template_dims = [
        "m",
        "n",
        "k",
        "num_blocks_m",
        "num_blocks_n",
        "num_blocks_k",
        "block_size_m",
        "block_size_n",
        "block_size_k",
    ]

    template_dtypes = [
        "a_type",
        "scale_type",
        "c_type",
    ]

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)  # Shape m, k
        b_desc = ksel.arg_tensor(1)  # Shape n, k
        as_desc = ksel.arg_tensor(2) # Shape num_bl_m, num_bl_k
        bs_desc = ksel.arg_tensor(3) # Shape num_bl_n, num_bl_k

        # a arg
        a_num_m, a_bs_m, a_num_k, a_bs_k = a_desc.t.shape 
        torch._check(
            a_desc.t.dtype.is_floating_point,
            lambda: f"mmt_block_scaled_f8 arg 'a': Expected floating point (got {a_desc.t.dtype})",
        )

        # b arg
        b_num_n, b_bs_n, b_num_k, b_bs_k = b_desc.t.shape
        torch._check(
            b_desc.t.dtype.is_floating_point,
            lambda: f"mmt_block_scaled_f8 arg 'b': Expected floating point (got {b_desc.t.dtype})",
        )

        # as arg
        as_num_m, as_one_m, as_num_k, as_one_k = as_desc.t.shape
        torch._check(
            (as_num_m == a_num_m) and (as_num_k == a_num_k),
            lambda: "mmt_block_scaled_f8 arg 'as': got incompatible number of blocks.\n"
             f"Scale : {as_desc.t.shape}, for quantized tensor :{a_desc.t.shape})",
        )
        torch._check(
            (as_one_m == 1) and (as_one_k == 1),
            lambda: f"mmt_block_scaled_f8 arg 'as': dims 1 and 3 should be 1 (got {as_desc.t.shape})",
        )

        # bs arg
        bs_num_n, bs_one_n, bs_num_k, bs_one_k = bs_desc.t.shape
        torch._check(
            (bs_num_n == b_num_n) and (bs_num_k == b_num_k),
            lambda: "mmt_block_scaled_f8 arg 'bs': got incompatible number of blocks.\n"
             f"Scale : {bs_desc.t.shape}, for quantized tensor :{b_desc.t.shape})",
        )
        torch._check(
            (bs_one_n == 1) and (bs_one_k == 1),
            lambda: f"mmt_block_scaled_f8 arg 'bs': dims 1 and 3 should be 1 (got {bs_desc.t.shape})",
        )

        # contracting dims:
        torch._check(
            (a_num_k == b_num_k) and (a_bs_k == b_bs_k),
            lambda: "mmt_block_scaled_f8 ars 'a' and 'b': incompatible contracting dimensions.\n"
            f"Got 'a' : {a_desc.t.shape} vs. 'b' : {b_desc.t.shape}.",
        )

        # Specialize on block sizes and numbers of blocks
        a_desc.specialize_all_dims()
        b_desc.specialize_all_dims()
        as_desc.specialize_all_dims()
        bs_desc.specialize_all_dims()

        # Shape m, n
        c_desc = ksel.return_new_tensor([a_num_m * a_bs_m, b_num_n * b_bs_n], dtype=torch.float32)
        c_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        a_tensor_type = RankedTensorType(a.type)
        b = kb.arg_value(1)
        b_tensor_type = RankedTensorType(b.type)
        a_s = kb.arg_value(2)
        a_s_tensor_type = RankedTensorType(a_s.type)
        b_s = kb.arg_value(3)
        b_s_tensor_type = RankedTensorType(b_s.type)

        num_m, bs_m, num_k, bs_k = a_tensor_type.shape
        bs_n = b_tensor_type.get_dim_size(1)
        num_m, one0, num_k, one1 = a_s_tensor_type.shape
        num_n = b_s_tensor_type.get_dim_size(0)
        m = num_m * bs_m
        n = num_n * bs_n
        k = num_k * bs_k
        a_type_str = str(a_tensor_type.element_type)
        scale_type_str = str(a_s_tensor_type.element_type)
        # hard-coded for now
        c_type_str = "f32"

        template_file = "mmt_block_scaled_f8_2d_expand.mlir"
        target_function_name = (
            f"sharktank_mmt_block_scaled_f8_2d_{m}_{bs_m}_{n}_{bs_n}_{k}_{bs_k}_{c_type_str}"
        )

        kwargs = dict(zip(self.template_dims, [m, n, k, num_m, num_n, num_k, bs_m, bs_n, bs_k]))
        kwargs.update(dict(zip(self.template_dtypes, [a_type_str, scale_type_str, c_type_str])))

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            **kwargs,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
