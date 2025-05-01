// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!q_type = tensor<{{b1}}x{{b2}}x{{l}}x{{d}}x{{i_dtype}}>
!k_type = tensor<{{b1}}x{{b2}}x{{s}}x{{d}}x{{i_dtype}}>
!v_type = tensor<{{b1}}x{{b2}}x{{s}}x{{e}}x{{i_dtype}}>
!a_type = tensor<{{l}}x{{s}}x{{a_dtype}}>
!trans_v_type = tensor<{{b1}}x{{b2}}x{{e}}x{{s}}x{{i_dtype}}>
!output_type = tensor<{{b1}}x{{b2}}x{{l}}x{{e}}x{{o_dtype}}>
!o_type = tensor<{{b1}}x{{b2}}x{{l}}x{{e}}xf32>
!o_dyn_type = tensor<?x?x?xf32>
!o_collapsed_type = tensor<{{b}}x{{l}}x{{e}}xf32>
!q_collapsed_type = tensor<{{b}}x{{l}}x{{d}}x{{i_dtype}}>
!k_collapsed_type = tensor<{{b}}x{{s}}x{{d}}x{{i_dtype}}>
!v_collapsed_type = tensor<{{b}}x{{s}}x{{e}}x{{i_dtype}}>
!a_collapsed_type = tensor<{{l}}x{{s}}x{{a_dtype}}>
!s_type = tensor<{{scale_dtype}}>

module {

util.func private @{{func_name}}(
    %q: !q_type,
    %k: !k_type,
    %v: !v_type,
    %s: !s_type,
    %a: !a_type,
    %p: !s_type) -> !output_type {

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %b0 = arith.constant {{b}} : index


        %l = tensor.dim %q, %c2 : !q_type
        %e = tensor.dim %v, %c3 : !v_type

        %scale = tensor.extract %s[] : !s_type
        %prob_output_scale = tensor.extract %p[] : !s_type
        %empty_dyn = tensor.empty(%b0, %l, %e) : !o_dyn_type
        %empty = tensor.cast %empty_dyn : !o_dyn_type to !o_collapsed_type

        %collapsed_q = tensor.collapse_shape %q [[0, 1], [2], [3]] : !q_type into !q_collapsed_type
        %collapsed_k = tensor.collapse_shape %k [[0, 1], [2], [3]] : !k_type into !k_collapsed_type
        %collapsed_v = tensor.collapse_shape %v [[0, 1], [2], [3]] : !v_type into !v_collapsed_type

        %atten = iree_linalg_ext.attention {indexing_maps = [
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2)>,
                    affine_map<(d0, d1, d2, d3, d4) -> ()>,
                    affine_map<(d0, d1, d2, d3, d4) -> ()>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]}
                    ins(%collapsed_q, %collapsed_k, %collapsed_v, %scale : !q_collapsed_type, !k_collapsed_type, !v_collapsed_type, {{scale_dtype}}) probOutputScale(%prob_output_scale : {{scale_dtype}}) mask(%a : !a_collapsed_type)  outs(%empty : !o_collapsed_type) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score : f32
                    } -> !o_collapsed_type
        %expanded_o = tensor.expand_shape %atten [[0,1], [2], [3]] output_shape [{{b1}}, {{b2}}, %l, %e] : !o_collapsed_type into !o_type
        %atten_trunc_empty = tensor.empty(%l) : !output_type
        %atten_trunc = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
            ins(%expanded_o : !o_type)
            outs(%atten_trunc_empty : !output_type) {
        ^bb0(%in: f32, %out: {{o_dtype}}):
            %trunc = arith.truncf %in : f32 to {{o_dtype}}
            linalg.yield %trunc : {{o_dtype}}
        } -> !output_type
        util.return %atten_trunc : !output_type
    }
}
