// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!elem_type = {{dtype}}
!tensor_type = {% for i in range(rank) %}?x{% endfor %}x!elem_type

module {

util.func private @sharktank_elementwise_tensor_tensor_{{op_str}}_{{rank}}_{{dtype}}(
    %x: !tensor_type, %y: !tensor_type)
    -> !tensor_type {
  {% for i in range(rank) %}
  %k{{i}} = arith.constant {{i}} : index
  %dim0 = tensor.dim %x, %k{{i}}: !x_tensor_type
  {% endfor %}
  %result_empty = tensor.empty({% for i in range(rank - 1) %}%dim{{i}}, {% endfor %}dim{{rank - 1}}) : !tensor_type
  %result = linalg.generic {
      indexing_maps = [
          affine_map<({% for i in range(rank - 1) %}%d{{i}}, {% endfor %}d{{rank - 1}}) -> ({% for i in range(rank - 1) %}%d{{i}}, {% endfor %}d{{rank - 1}})>,
          affine_map<({% for i in range(rank - 1) %}%d{{i}}, {% endfor %}d{{rank - 1}}) -> ({% for i in range(rank - 1) %}%d{{i}}, {% endfor %}d{{rank - 1}})>
      ],
      iterator_types = [
        "{% for i in range(rank - 1) %}"parallel", {% endfor %}"parallel"
      ] }
      ins(%x, %y : !tensor_type, !tensor_type)
      outs(%result_empty : !accum_tensor_type) {
  ^bb0(%x_element: !elem_type, %y_element: !elem_type, %out: !elem_type):
      %elem_result = arith.{{op_str}}f %x_element, %y_element : !elem_type
      linalg.yield %bmm_accum : !elem_type
  } -> !tensor_type

  util.return %result : !tensor_type
}

}
