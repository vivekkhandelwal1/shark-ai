# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
import pytest

from typing import Any
from unittest.mock import patch

import shortfin.array as sfnp

from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
)
from shortfin_apps.llm.components.token_selection_strategy import (
    build_token_selector_config,
    GreedyTokenSelectionStrategy,
    TokenSelectionStrategy,
    DecodeConfig,
)
from shortfin_apps.llm.components.token_selection_strategy.greedy_token_selection_strategy import (
    GreedyBeam,
)

logger = logging.getLogger(__name__)


class FakeBatcher:
    def __init__(self, submit_cb, workitem_cb):
        self.submit = submit_cb
        self.reserve_workitem = workitem_cb
        self.complete_workitem = workitem_cb


@pytest.fixture(scope="function")
def greedy_token_selection_strategy():
    yield GreedyTokenSelectionStrategy(
        None,
    )


@pytest.fixture(scope="function")
def greedy_beam(exec_req):
    yield GreedyBeam(
        exec_req,
    )


def batcher_workitem_cb(_: int):
    pass


def approximately_equal(a: Any, b: Any, rel_tol=1e-2, abs_tol=0.0) -> bool:
    """
    Recursively checks if two nested lists (or scalar values) are approximately equal.

    Args:
        a: First list or scalar.
        b: Second list or scalar.
        rel_tol: Relative tolerance.
        abs_tol: Absolute tolerance.

    Returns:
        True if all corresponding elements are approximately equal.
    """
    # If both are lists, iterate element-wise
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(
            approximately_equal(sub_a, sub_b, rel_tol, abs_tol)
            for sub_a, sub_b in zip(a, b)
        )

    # Otherwise, assume they are scalars and compare
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def test_greedy_beam_sample_logits(device, greedy_beam):
    greedy_beam.temperature = 1.0

    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    greedy_beam.exec_req.result_logits = src
    token = greedy_beam.sample_logits()
    assert token == 15

    data[10] = 42.0
    src.items = data
    greedy_beam.exec_req.result_logits == src
    token = greedy_beam.sample_logits()
    assert token == 10


def test_greedy_update_exec_req(greedy_beam):
    last_token = 42
    expected_start_position = greedy_beam.exec_req.start_position + 1

    greedy_beam.last_token = last_token
    greedy_beam.update_exec_req()

    assert greedy_beam.exec_req.input_token_ids[-1] == last_token
    assert greedy_beam.exec_req.start_position == expected_start_position


@pytest.mark.asyncio
async def test_greedy_decode_single(
    device, exec_req: LlmInferenceExecRequest, greedy_token_selection_strategy
):
    def _batcher_callback(request: LlmInferenceExecRequest):
        """Mock the batcher function to isolate `TokenSelectionStrategy.prefill`.

        This adds a `device_array` to the `LlmInferenceExecRequest's` result_logits.
        Then we set the request to done, effectively simulating what would
        happen under the hood.

        Args:
            request (LlmInferenceExecRequest): Request that would be submitted to batcher.
        """
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()

    results_array = []

    def _results_callback(token: int):
        results_array.append(token)

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.GREEDY,
        max_completion_tokens=1,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(_batcher_callback, batcher_workitem_cb),
        decode_batcher=FakeBatcher(_batcher_callback, batcher_workitem_cb),
        results_callback=_results_callback,
        eos_token_id=-1,
    )

    # Single token generated
    with patch.object(
        greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        await greedy_token_selection_strategy.decode(exec_req)

        assert results_array[0] == 15
        assert exec_req.input_token_ids[-1] == 15
        assert exec_req.start_position == 6


@pytest.mark.asyncio
async def test_greedy_decode_multiple_completions(
    device, exec_req: LlmInferenceExecRequest, greedy_token_selection_strategy
):
    results_array = []

    def _results_callback(token: int):
        results_array.append(token)

    count = 0

    def _batcher_callback_multiple_completions(request: LlmInferenceExecRequest):
        """Mock the batcher function to isolate `TokenSelectionStrategy.prefill`.

        This adds a `device_array` to the `LlmInferenceExecRequest's` result_logits.
        Then we set the request to done, effectively simulating what would
        happen under the hood.

        Args:
            request (LlmInferenceExecRequest): Request that would be submitted to batcher.
        """
        nonlocal count
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]

        # Set max to an explicit index
        data[count] = 16
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()
        count += 1

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.GREEDY,
        max_completion_tokens=5,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(
            _batcher_callback_multiple_completions, batcher_workitem_cb
        ),
        decode_batcher=FakeBatcher(
            _batcher_callback_multiple_completions, batcher_workitem_cb
        ),
        results_callback=_results_callback,
        eos_token_id=-1,
    )

    # Multiple tokens generated
    with patch.object(
        greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        await greedy_token_selection_strategy.decode(exec_req)

        assert results_array == [0, 1, 2, 3, 4]
        assert len(exec_req.input_token_ids) == 11
        assert exec_req.start_position == 10


@pytest.mark.asyncio
async def test_greedy_decode_eos_token(
    device, exec_req: LlmInferenceExecRequest, greedy_token_selection_strategy
):
    results_array = []

    def _results_callback(token: int):
        results_array.append(token)

    count = 0

    def _batcher_callback_multiple_completions(request: LlmInferenceExecRequest):
        """Mock the batcher function to isolate `TokenSelectionStrategy.prefill`.

        This adds a `device_array` to the `LlmInferenceExecRequest's` result_logits.
        Then we set the request to done, effectively simulating what would
        happen under the hood.

        Args:
            request (LlmInferenceExecRequest): Request that would be submitted to batcher.
        """
        nonlocal count
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]

        # Set max to an explicit index
        data[count] = 16
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()
        count += 1

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.GREEDY,
        max_completion_tokens=10,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(
            _batcher_callback_multiple_completions, batcher_workitem_cb
        ),
        decode_batcher=FakeBatcher(
            _batcher_callback_multiple_completions, batcher_workitem_cb
        ),
        results_callback=_results_callback,
        eos_token_id=5,
    )

    # Multiple tokens generated, eos is hit
    with patch.object(
        greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        await greedy_token_selection_strategy.decode(exec_req)

        assert results_array == [0, 1, 2, 3, 4, 5]
        assert len(exec_req.input_token_ids) == 11
        assert exec_req.start_position == 10
