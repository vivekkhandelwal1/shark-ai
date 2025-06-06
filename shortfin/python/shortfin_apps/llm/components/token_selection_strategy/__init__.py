# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, List, Union

from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)

from .config import (
    DecodeConfig,
    TokenSelectionStrategy,
    get_strategy_from_str,
)
from .beam_search_token_selection_strategy import BeamSearchTokenSelectionStrategy
from .independent_token_selection_strategy import IndependentTokenSelectionStrategy
from .sampler import Sampler


def build_token_selector_config(
    decode_config: DecodeConfig,
    prefill_batcher,
    decode_batcher,
    results_callback: Callable[[Union[int, List[int]]], None],
    eos_token_id: int,
) -> TokenSelectionStrategyConfig:
    """Build a configuration class for a given token selection strategy.

    Args:
        token_selection_strategy (TokenSelectionStrategy): Strategy to use.
        prefill_callback (Callable[[LlmInferenceExecRequest], None]): Callback for invoking prefill. Typically a batcher function.
        decode_callback (Callable[[LlmInferenceExecRequest], None]): Callback for invoking decode. Typically a batcher function.
        results_callback (Callable[[Union[int, List[int]]], None]): Callback for during or after tokens are generated, depending on the strategy.
        eos_token_id (int): Token to stop generation on.
        max_completion_tokens (int): Max tokens to generate.

    Raises:
        NotImplementedError: Unsupported `TokenSelectionStrategy`.

    Returns:
        TokenSelectionStrategyConfig: Instantiated config for token selector.
    """
    if decode_config.token_selection_strategy not in {
        strategy for strategy in TokenSelectionStrategy
    }:
        raise NotImplementedError(
            f"Unsupported token selection strategy: {decode_config.token_selection_strategy}.\n"
            f"Supported strategies: {','.join([strategy.name for strategy in TokenSelectionStrategy])}"
        )
    return TokenSelectionStrategyConfig(
        decode_config,
        prefill_callback=prefill_batcher.submit,
        decode_callback=decode_batcher.submit,
        decode_begin_callback=decode_batcher.reserve_workitem,
        decode_end_callback=decode_batcher.complete_workitem,
        results_callback=results_callback,
        eos_token_id=eos_token_id,
    )


def build_token_selector(
    config: TokenSelectionStrategyConfig,
) -> BaseTokenSelectionStrategy:
    """Build a token selector, given a strategy and a config.

    Args:
        token_selection_strategy (TokenSelectionStrategy): Strategy to use.
        config (TokenSelectionStrategyConfig): Config containing necessary parameters for execution.

    Raises:
        NotImplementedError: Unsupported `TokenSelectionStrategy`.

    Returns:
        BaseTokenSelectionStrategy: Instantiated token selector. Either `IndependentTokenSelectionStrategy` or `BeamSearchTokenSelectionStrategy`.
    """
    strategy_map = {
        TokenSelectionStrategy.INDEPENDENT: IndependentTokenSelectionStrategy,
        TokenSelectionStrategy.BEAM_SEARCH: BeamSearchTokenSelectionStrategy,
    }
    if config.decode_config.token_selection_strategy not in strategy_map:
        raise NotImplementedError(
            f"Unsupported token selection strategy: {config.decode_config.token_selection_strategy}.\n"
            f"Supported strategies: {','.join([strategy.name for strategy in TokenSelectionStrategy])}"
        )

    return strategy_map[config.decode_config.token_selection_strategy](
        token_selection_strategy_config=config
    )


def is_multi_response(decode_config: DecodeConfig) -> bool:
    strategy = decode_config.token_selection_strategy
    num_beams = decode_config.num_beams

    return strategy == TokenSelectionStrategy.BEAM_SEARCH or (
        strategy == TokenSelectionStrategy.INDEPENDENT and num_beams > 1
    )


__all__ = [
    "BaseTokenSelectionStrategy",
    "TokenSelectionStrategyConfig",
    "TokenSelectionStrategy",
    "Sampler",
    "BeamSearchTokenSelectionStrategy",
    "IndependentTokenSelectionStrategy",
    "build_token_selector",
    "build_token_selector_config",
    "get_strategy_from_str",
    "is_multi_response",
]
