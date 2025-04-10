# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest common_test.py
"""

import pytest
from . import common

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
from iree.compiler.dialects import transform  # type: ignore
from iree.compiler.dialects import _builtin_ops_gen  # type: ignore

from .test_utils import tuner_ctx
from .test_utils import mlir_ctx


def test_get_shaped_type_element_bitwidth(tuner_ctx: common.TunerContext) -> None:
    assert common.ShapedType([1024, 2048], tuner_ctx.type.i8).bitwidth == 8
    assert common.ShapedType([2048], tuner_ctx.type.i32).bitwidth == 32
    assert common.ShapedType([2048, 512, 384], tuner_ctx.type.f8E4M3FNUZ).bitwidth == 8
    assert common.ShapedType([1, 1], tuner_ctx.type.f16).bitwidth == 16


def test_get_shaped_type_to_str(tuner_ctx: common.TunerContext) -> None:
    assert str(common.ShapedType([1024, 2048], tuner_ctx.type.i8)) == "1024x2048xi8"
    assert str(common.ShapedType([1024], tuner_ctx.type.f32)) == "1024xf32"
    assert str(common.ShapedType([1, 2, 3], tuner_ctx.type.f16)) == "1x2x3xf16"
    assert str(common.ShapedType([-1, 2, 3], tuner_ctx.type.f16)) == "?x2x3xf16"


def test_gpu_pipeline_options(tuner_ctx: common.TunerContext) -> None:
    options = iree_gpu.PipelineOptionsAttr.get()
    assert str(options) == "#iree_gpu.pipeline_options<>"

    options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
    assert str(options) == "#iree_gpu.pipeline_options<prefetch_shared_memory = true>"

    options = iree_gpu.PipelineOptionsAttr.get(
        prefetch_shared_memory=True, no_reduce_shared_memory_bank_conflicts=False
    )
    assert (
        str(options)
        == "#iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>"
    )

    options = iree_gpu.PipelineOptionsAttr.get(
        reorder_workgroups_strategy=iree_gpu.ReorderWorkgroupsStrategyAttr.get(
            iree_gpu.ReorderWorkgroupsStrategy.Transpose
        )
    )
    assert (
        str(options)
        == "#iree_gpu.pipeline_options<reorder_workgroups_strategy = <Transpose>>"
    )


def test_get_pipeline_config(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[4, 8, 0],
        reduction=[0, 0, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 32, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    config1_str: str = str(
        compilation_info.translation_info.configuration[common.LLVM_FUNC_ATTRS_KEY]
    )
    assert config1_str == '{"amdgpu-waves-per-eu" = "2"}'

    pipeline_options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
    config_dict = common.get_translation_info_config(pipeline_options, 4)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 32, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    config2_str: str = str(compilation_info.translation_info.configuration)
    assert (
        config2_str
        == '{gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}}'
    )


def test_get_compatible_mfma_intrinsics(tuner_ctx: common.TunerContext) -> None:
    assert common.get_compatible_mfma_intrinsics(
        common.ProblemSize(
            common.ContractionSizes([2048], [1280], [1280]),
            common.ShapedType([2048, 1280], tuner_ctx.type.f16),
            common.ShapedType([1280, 1280], tuner_ctx.type.f16),
            common.ShapedType([2048, 1280], tuner_ctx.type.f32),
            common.DispatchKind.contraction,
            common.ContractionDimensions([0], [1], [2]),
        ),
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    ) == [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
    ]

    assert common.get_compatible_mfma_intrinsics(
        common.ProblemSize(
            common.ContractionSizes([2048], [1280], [1280]),
            common.ShapedType([2048, 1280], tuner_ctx.type.i8),
            common.ShapedType([1280, 1280], tuner_ctx.type.i8),
            common.ShapedType([2048, 1280], tuner_ctx.type.i32),
            common.DispatchKind.contraction,
            common.ContractionDimensions([0], [1], [2]),
        ),
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    ) == [
        iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
        iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
    ]

    assert (
        common.get_compatible_mfma_intrinsics(
            common.ProblemSize(
                common.ContractionSizes([968], [320], [640], [64]),
                common.ShapedType([64, 968, 640], tuner_ctx.type.f32),
                common.ShapedType([64, 640, 320], tuner_ctx.type.f32),
                common.ShapedType([64, 968, 320], tuner_ctx.type.f32),
                common.DispatchKind.contraction,
                common.ContractionDimensions([1], [2], [3], [0]),
            ),
            [
                iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
                iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
                iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
                iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
            ],
        )
        == []
    )


def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        workgroup=[4, 8, 0],
        reduction=[0, 0, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
    )

    assert (
        str(lowering_config)
        == "#iree_gpu.lowering_config<{reduction = [0, 0, 16], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 8, 0]}>"
    )

    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 32, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    assert compilation_info.lowering_config.mma_kind is None
    assert compilation_info.lowering_config.subgroup_count_mn == (1, 1)


def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    first_module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence } {
        }
    """

    second_module_str = """
        module @inner_module_b
            attributes { transform.with_named_sequence } {
        }
    """

    first_ir_module = ir.Module.parse(first_module_str, context)
    second_ir_module = ir.Module.parse(second_module_str, context)

    module = common.combine_tuning_specs(tuner_ctx, [first_ir_module, second_ir_module])
    assert module
    assert "transform.with_named_sequence" in module.operation.attributes

    inner_ops = list(module.body.operations)
    assert all(
        isinstance(op, _builtin_ops_gen.ModuleOp) for op in inner_ops
    ), "Not all ops are builtin.module"
    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
    assert (
        inner_ops[0].sym_name.value == "inner_module_a"
    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
    assert (
        inner_ops[1].sym_name.value == "inner_module_b"
    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"


def test_link_tuning_specs(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg : !transform.any_op
            }

            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
                    : (!transform.any_op) -> (!transform.any_op)
                transform.yield %res : !transform.any_op
            }
        }
    """

    ir_module = ir.Module.parse(module_str, context)
    linked_module = common.link_tuning_specs(tuner_ctx, [ir_module])
    assert (
        linked_module is ir_module
    ), "Expected single input module to be returned without modification"

    first_ir_module = ir.Module.parse(module_str, context)
    second_ir_module = ir.Module.parse(module_str, context)
    second_ir_module.operation.attributes["sym_name"] = ir.StringAttr.get(
        "inner_module_b"
    )
    linked_module = common.link_tuning_specs(
        tuner_ctx, [first_ir_module, second_ir_module]
    )
    assert linked_module

    assert "transform.with_named_sequence" in linked_module.operation.attributes
    assert (
        "iree_codegen.tuning_spec_with_default_entrypoint"
        in linked_module.operation.attributes
    )

    inner_ops = list(linked_module.body.operations)
    # Check that inner modules have been merged into the top-level module and no inner modules remain.
    assert all(
        not isinstance(op, _builtin_ops_gen.ModuleOp) for op in inner_ops
    ), "Unexpected inner builtin.module ops found"

    named_sequences = []
    kernel_config_op = None
    for op in linked_module.body.operations:
        if isinstance(op, transform.NamedSequenceOp):
            sym_name_attr = op.sym_name
            assert sym_name_attr is not None
            named_sequences.append(sym_name_attr.value)
            if sym_name_attr.value == "__kernel_config":
                kernel_config_op = op

    assert kernel_config_op is not None, "Missing @__kernel_config"


def test_link_tuning_specs_raises_error(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence } {
        }
    """

    module = ir.Module.parse(module_str, context)
    module.operation.attributes[
        "iree_codegen.tuning_spec_with_default_entrypoint"
    ] = ir.UnitAttr.get()
    with pytest.raises(RuntimeError) as exc_info:
        common.link_tuning_specs(tuner_ctx, [module, module])
        # iree-opt should fail due to missing named sequence @__kernel_config entrypoint required
        # by the `iree_codegen.tuning_spec_with_default_entrypoint` attribute.
        assert "iree-opt failed" in str(exc_info.value)


def test_get_matcher_names_from_td_spec(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module attributes { transform.with_named_sequence } {
            transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @match_foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @match_bar(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op ) -> !transform.any_op
                attributes { iree_codegen.tuning_spec_entrypoint } {
                %0 = transform.foreach_match in %arg0
                    @match_foo -> @apply_op_config,
                    @match_bar -> @apply_op_config : (!transform.any_op) -> !transform.any_op
                transform.yield %0 : !transform.any_op
            }
        }
    """

    module = ir.Module.parse(module_str, context)
    matcher_names = common.get_matcher_names_from_td_spec(module)

    assert matcher_names == {"match_foo", "match_bar"}

    module_str = """
        module attributes { transform.with_named_sequence } {
            transform.named_sequence @__kernel_config(%arg0: !transform.any_op) -> !transform.any_op
                attributes { iree_codegen.tuning_spec_entrypoint } {
                transform.yield %arg0 : !transform.any_op
            }
        }
    """
    module = ir.Module.parse(module_str, context)
    matcher_names = common.get_matcher_names_from_td_spec(module)
    assert matcher_names == set()


def test_get_matcher_overlap_info(tuner_ctx: common.TunerContext) -> None:
    starter = {"match_a", "match_b", "match_c"}
    current = {"match_b", "match_d"}

    overlapping, unique = common.get_matcher_overlap_info(starter, current)

    assert overlapping == {"match_b"}
    assert unique == {"match_a", "match_c"}

    starter = {"match_x", "match_y"}
    current = {"match_a", "match_b"}
    overlapping, unique = common.get_matcher_overlap_info(starter, current)
    assert overlapping == set()
    assert unique == {"match_x", "match_y"}

    starter = {"match_a", "match_b"}
    current = {"match_a", "match_b", "match_c"}
    overlapping, unique = common.get_matcher_overlap_info(starter, current)
    assert overlapping == {"match_a", "match_b"}
    assert unique == set()


def test_determine_td_specs_to_link(
    tuner_ctx: common.TunerContext, caplog: pytest.LogCaptureFixture
) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module attributes { transform.with_named_sequence } {
            transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @match_foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @match_bar(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op ) -> !transform.any_op
                attributes { iree_codegen.tuning_spec_entrypoint } {
                %0 = transform.foreach_match in %arg0
                    @match_foo -> @apply_op_config,
                    @match_bar -> @apply_op_config : (!transform.any_op) -> !transform.any_op
                transform.yield %0 : !transform.any_op
            }
        }
    """
    starter_td_spec = ir.Module.parse(module_str, context)
    current_td_spec = ir.Module.parse(module_str, context)

    td_specs_to_link = common.determine_td_specs_to_link(
        [current_td_spec, starter_td_spec],
        log_duplicates=True,
    )

    assert td_specs_to_link == [current_td_spec]
    assert "match_foo" in caplog.text
    assert "match_bar" in caplog.text
    assert "already been tuned in the starter" in caplog.text

    caplog.clear()
    module_str = """
        module attributes { transform.with_named_sequence } {
            transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @match_baz(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op ) -> !transform.any_op
                attributes { iree_codegen.tuning_spec_entrypoint } {
                %0 = transform.foreach_match in %arg0
                    @match_baz -> @apply_op_config : (!transform.any_op) -> !transform.any_op
                transform.yield %0 : !transform.any_op
            }
        }
    """
    current_td_spec = ir.Module.parse(module_str, context)
    td_specs_to_link = common.determine_td_specs_to_link(
        [starter_td_spec, current_td_spec],
        log_duplicates=True,
    )

    assert td_specs_to_link == [starter_td_spec, current_td_spec]
    assert "match_baz" not in caplog.text
    assert "already been tuned" not in caplog.text
