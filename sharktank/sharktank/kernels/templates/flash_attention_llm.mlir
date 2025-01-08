// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!q_type = tensor<{{b1}}x{{b2}}x{{l}}x{{d}}x{{i_dtype}}>
!k_type = tensor<{{b1}}x{{b2}}x{{s}}x{{d}}x{{i_dtype}}>
!v_type = tensor<{{b1}}x{{b2}}x{{s}}x{{e}}x{{i_dtype}}>
!a_type = tensor<{{b1}}x{{b2}}x{{l}}x{{s}}x{{i_dtype}}>
!trans_v_type = tensor<{{b1}}x{{b2}}x{{e}}x{{s}}x{{i_dtype}}>
!o_type = tensor<{{b1}}x{{b2}}x{{l}}x{{e}}x{{o_dtype}}>
!o_dyn_type = tensor<?x?x?x?x{{o_dtype}}>
!s_type = tensor<{{scale_dtype}}>

module {

util.func private @sharktank_llm_flash_attention_{{b1}}_{{b2}}_{{d}}_{{e}}_{{i_dtype}}_{{scale_dtype}}_{{o_dtype}}(
    %q: !q_type,
    %k: !k_type,
    %v: !v_type,
    %s: !s_type,
    %a: !a_type) -> !o_type {

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index

        %b0 = tensor.dim %q, %c0 : !q_type
        %b1 = tensor.dim %q, %c1 : !q_type

        %l = tensor.dim %q, %c2 : !q_type
        %e = tensor.dim %v, %c3 : !v_type

        %scale = tensor.extract %s[] : !s_type

        %empty_dyn = tensor.empty(%b0, %b1, %l, %e) : !o_dyn_type
        %empty = tensor.cast %empty_dyn : !o_dyn_type to !o_type

        %atten = iree_linalg_ext.attention {indexing_maps = [
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>]}
                    ins(%q, %k, %v, %scale, %a : !q_type, !k_type, !v_type, {{scale_dtype}}, !a_type) outs(%empty : !o_type) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score : f32
                    } -> !o_type
        util.return %atten : !o_type
    }
}
