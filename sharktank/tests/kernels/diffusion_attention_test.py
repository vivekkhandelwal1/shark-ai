# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
from parameterized import parameterized

import torch

from iree.turbine import aot
from sharktank import kernels
from sharktank.types import layout_utils


class punet_attention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(420)

    @parameterized.expand(
        [
            (torch.float32, 1e-3, 1e-4),
            (torch.float16, 1e-3, 1e-4),
        ]
    )
    def test_compare_torch_spda(self, dtype, atol, rtol):
        H = 4  # Head dim
        N = 1  # Batch Size
        L = 7  # Target Seq Len
        S = 6  # Source Seq Len
        Eqk = Ev = 64  # embedding dimensions with subscript identifiers

        q = torch.rand([N, H, L, Eqk], dtype=dtype)
        k = torch.rand([N, H, S, Eqk], dtype=dtype)
        v = torch.rand([N, H, S, Ev], dtype=dtype)
        # mask is same type as inputs, therefore its added to score
        mask = torch.rand([N, H, L, S], dtype=dtype)
        scale = torch.tensor(1.0, dtype=dtype)
        result = kernels.flash_attention(q, k, v, mask, scale)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, mask, scale=scale
        )

        torch.testing.assert_close(result.to(dtype), ref, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            (torch.float32, 1e-3, 1e-4),
            # (torch.float16, 1e-3, 1e-4),
        ]
    )
    def test_export_dynamic(self, dtype, atol, rtol):
        H = 4  # Head dim
        N = 1  # Batch Size
        L = 19  # Target Seq Len
        S = 19  # Source Seq Len
        Eqk = Ev = 64  # embedding dimensions with subscript identifiers

        q = torch.rand([N, H, L, Eqk], dtype=dtype)
        k = torch.rand([N, H, S, Eqk], dtype=dtype)
        v = torch.rand([N, H, S, Ev], dtype=dtype)
        # mask is same type as inputs, therefore its added to score
        mask = torch.rand([N, H, L, S], dtype=dtype)
        scale = torch.tensor(1.0, dtype=dtype)
        L_dim = torch.export.Dim("L")
        S_dim = torch.export.Dim("S")

        class MyModule(torch.nn.Module):
            def forward(self, q, k, v, mask, scale):
                return kernels.flash_attention(q, k, v, mask, scale)

        mod = MyModule()
        dtype = torch.dtype
        ep = torch.export.export(
            mod,
            args=(q, k, v, mask, scale),
            dynamic_shapes={
                "q": {2: L_dim},
                "k": {2: S_dim},
                "v": {2: S_dim},
                "mask": {2: L_dim, 3: S_dim},
                "scale": {},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        # self.assertIn("@sharktank_batch_matmul_transpose_b_L4x16x2xi32_R4x8x2xi32", asm)
        print(asm)
        with open("sample.mlir", "w+") as f:
            f.write(asm)


#    @parameterized.expand(
#        [
#            (torch.float32, torch.float32, torch.float32, 1e-2, 1e-3),
#            (torch.float32, torch.float16, torch.float32, 1e-2, 1e-3),
#            (torch.float16, torch.float16, torch.float32, 1e-2, 1e-3),
#        ]
#    )
#    def test_basic_mek_menk_men(self, a_dtype, d_dtype, ref_dtype, atol, rtol):
#        a = torch.rand([2, 4, 320], dtype=a_dtype) / 256.0
#        d = torch.rand([2, 4, 8, 10, 1], dtype=d_dtype) / 256.0
#        qs = (torch.rand([2, 4, 8, 10, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
#        m = torch.rand([2, 4, 8, 10, 1], dtype=d_dtype) + 16.0
#        einsum_string = "mek,menk->men"
#        result = kernels.einsum_2args_q4(a, d, qs, m, einsum_string)
#
#        # Dequantize and test with normal matmul.
#        # Tolerances are empirical and results are not expected to match exactly.
#        qs_i8 = layout_utils.promote_linear_i4_block_to_i8(qs)
#        b = (d.to(ref_dtype) * qs_i8.to(ref_dtype) + m.to(ref_dtype)).flatten(3)
#        ref = torch.einsum(einsum_string, a.to(ref_dtype), b.to(ref_dtype))
#        torch.testing.assert_close(result.to(ref_dtype), ref, atol=atol, rtol=rtol)
#
#    @parameterized.expand(
#        [
#            (torch.float32, torch.float32, torch.float32, 1e-2, 1e-3),
#            (torch.float32, torch.float16, torch.float32, 1e-2, 1e-3),
#            (torch.float16, torch.float16, torch.float32, 1e-2, 1e-3),
#        ]
#    )
#    def test_basic_me_men_men(self, a_dtype, d_dtype, ref_dtype, atol, rtol):
#        a = torch.rand([2, 4], dtype=a_dtype) / 256.0
#        d = torch.rand([2, 4, 10, 1], dtype=d_dtype) / 256.0
#        qs = (torch.rand([2, 4, 10, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
#        m = torch.rand([2, 4, 10, 1], dtype=d_dtype) + 16.0
#        einsum_string = "me,men->men"
#        result = kernels.einsum_2args_q4(a, d, qs, m, einsum_string)
#
#        # Dequantize and test with normal matmul.
#        # Tolerances are empirical and results are not expected to match exactly.
#        qs_i8 = layout_utils.promote_linear_i4_block_to_i8(qs)
#        b = (d.to(ref_dtype) * qs_i8.to(ref_dtype) + m.to(ref_dtype)).flatten(2)
#        ref = torch.einsum(einsum_string, a.to(ref_dtype), b.to(ref_dtype))
#        torch.testing.assert_close(result.to(ref_dtype), ref, atol=atol, rtol=rtol)
#
#    def testExportDynamicDims(self):
#        class MyModule(torch.nn.Module):
#            def forward(self, a, d, qs, m):
#                return kernels.einsum_2args_q4(a, d, qs, m, "ij,jk->ik")
#
#        mod = MyModule()
#        ep = torch.export.export(
#            mod,
#            args=(
#                torch.rand([16, 320], dtype=torch.float32),
#                torch.rand([320, 2, 1], dtype=torch.float16),
#                (torch.rand([320, 2, 16], dtype=torch.float32) * 32).to(torch.uint8),
#                torch.rand([320, 2, 1], dtype=torch.float16),
#            ),
#            dynamic_shapes={
#                "a": {},
#                "d": {},
#                "qs": {},
#                "m": {},
#            },
#        )
#        output = aot.export(ep)
#        output.verify()
#        asm = str(output.mlir_module)
#        self.assertIn("@sharktank_einsum_2args_q4_ij_jk_ik_32_f32", asm)
#
#    def testExportStaticDims(self):
#        class MyModule(torch.nn.Module):
#            def forward(self, a, d, qs, m):
#                return kernels.einsum_2args_q4(a, d, qs, m, "mek,menk->men")
#
#        mod = MyModule()
#        ep = torch.export.export(
#            mod,
#            args=(
#                torch.rand([4, 16, 320], dtype=torch.float32),
#                torch.rand([4, 16, 2, 10, 1], dtype=torch.float16),
#                (torch.rand([4, 16, 2, 10, 16], dtype=torch.float32) * 32).to(
#                    torch.uint8
#                ),
#                torch.rand([4, 16, 2, 10, 1], dtype=torch.float16),
#            ),
#        )
#        output = aot.export(ep)
#        output.verify()
#        asm = str(output.mlir_module)
#        self.assertIn("@sharktank_einsum_2args_q4_mek_menk_men_32_f32", asm)
#

if __name__ == "__main__":
    unittest.main()
