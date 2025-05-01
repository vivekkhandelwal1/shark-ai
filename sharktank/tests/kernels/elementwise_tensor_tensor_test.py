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

from sharktank import kernels
from sharktank.types import PlanarQuantizedTensor, TensorScaledLayout


class elementwise_tensor_tensor_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, 1e-2, 1e-3),
            (torch.float32, torch.float32, 1e-2, 1e-3),
            (torch.float16, torch.float32, 1e-2, 1e-3),
        ]
    )
    def test_mul(self, dtype, ref_dtype, atol, rtol):
        shape = [17, 32]
        x = torch.rand(shape, dtype=dtype)
        y = torch.rand(shape, dtype=dtype)
        one = torch.tensor(1)
        x_layout = TensorScaledLayout(shape=shape, qs=x, d=one)
        x_qtensor = PlanarQuantizedTensor(shape=shape, layout=x_layout)
        y_layout = TensorScaledLayout(shape=shape, qs=y, d=one)
        y_qtensor = PlanarQuantizedTensor(shape=shape, layout=y_layout)
        result = kernels.elementwise_tensor_tensor(torch.mul, x_qtensor, y_qtensor)

        ref = x * y
        torch.testing.assert_close(result.to(ref_dtype), ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
