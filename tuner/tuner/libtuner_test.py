# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import math
from unittest.mock import call, patch, MagicMock
from . import libtuner

"""
Usage: python -m pytest libtuner_test.py
"""


def test_find_collisions() -> None:
    input = [(1, "abc"), (2, "def"), (3, "abc")]
    assert libtuner.find_collisions(input) == (True, [("abc", [1, 3]), ("def", [2])])
    input = [(1, "abc"), (2, "def"), (3, "hig")]
    assert libtuner.find_collisions(input) == (
        False,
        [("abc", [1]), ("def", [2]), ("hig", [3])],
    )


def test_collision_handler() -> None:
    input = [(1, "abc"), (2, "def"), (3, "abc"), (4, "def"), (5, "hig")]
    assert libtuner.collision_handler(input) == (True, [1, 2, 5])
    input = [(1, "abc"), (2, "def"), (3, "hig")]
    assert libtuner.collision_handler(input) == (False, [])


def test_extract_driver_names() -> None:
    user_devices = ["hip://0", "local-sync://default", "cuda://default"]
    expected_output = {"hip", "local-sync", "cuda"}

    assert libtuner.extract_driver_names(user_devices) == expected_output


def test_fetch_available_devices_success() -> None:
    drivers = ["hip", "local-sync", "cuda"]
    mock_devices = {
        "hip": [{"path": "ABCD", "device_id": 1}],
        "local-sync": [{"path": "default", "device_id": 2}],
        "cuda": [{"path": "default", "device_id": 3}],
    }

    with patch(f"{libtuner.__name__}.ireert.get_driver") as mock_get_driver:
        mock_driver = MagicMock()

        def get_mock_driver(name):
            mock_driver.query_available_devices.side_effect = lambda: mock_devices[name]
            return mock_driver

        mock_get_driver.side_effect = get_mock_driver

        actual_output = libtuner.fetch_available_devices(drivers)
        expected_output = [
            "hip://ABCD",
            "hip://0",
            "local-sync://default",
            "local-sync://1",
            "cuda://default",
            "cuda://2",
        ]

        assert actual_output == expected_output


def test_fetch_available_devices_failure() -> None:
    drivers = ["hip", "local-sync", "cuda"]
    mock_devices = {
        "hip": [{"path": "ABCD", "device_id": 1}],
        "local-sync": ValueError("Failed to initialize"),
        "cuda": [{"path": "default", "device_id": 1}],
    }

    with patch(f"{libtuner.__name__}.ireert.get_driver") as mock_get_driver:
        with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
            mock_driver = MagicMock()

            def get_mock_driver(name):
                if isinstance(mock_devices[name], list):
                    mock_driver.query_available_devices.side_effect = (
                        lambda: mock_devices[name]
                    )
                else:
                    mock_driver.query_available_devices.side_effect = lambda: (
                        _ for _ in ()
                    ).throw(mock_devices[name])
                return mock_driver

            mock_get_driver.side_effect = get_mock_driver

            actual_output = libtuner.fetch_available_devices(drivers)
            expected_output = ["hip://ABCD", "hip://0", "cuda://default", "cuda://0"]

            assert actual_output == expected_output
            mock_handle_error.assert_called_once_with(
                condition=True,
                msg="Could not initialize driver local-sync: Failed to initialize",
                error_type=ValueError,
                exit_program=True,
            )


def test_parse_devices() -> None:
    user_devices_str = "hip://0, local-sync://default, cuda://default"
    expected_output = ["hip://0", "local-sync://default", "cuda://default"]

    with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
        actual_output = libtuner.parse_devices(user_devices_str)
        assert actual_output == expected_output

        mock_handle_error.assert_not_called()


def test_parse_devices_with_invalid_input() -> None:
    user_devices_str = "hip://0, local-sync://default, invalid_device, cuda://default"
    expected_output = [
        "hip://0",
        "local-sync://default",
        "invalid_device",
        "cuda://default",
    ]

    with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
        actual_output = libtuner.parse_devices(user_devices_str)
        assert actual_output == expected_output

        mock_handle_error.assert_called_once_with(
            condition=True,
            msg=f"Invalid device list: {user_devices_str}. Error: {ValueError()}",
            error_type=argparse.ArgumentTypeError,
        )


def test_validate_devices() -> None:
    user_devices = ["hip://0", "local-sync://default"]
    user_drivers = {"hip", "local-sync"}

    with patch(f"{libtuner.__name__}.extract_driver_names", return_value=user_drivers):
        with patch(
            f"{libtuner.__name__}.fetch_available_devices",
            return_value=["hip://0", "local-sync://default"],
        ):
            with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
                libtuner.validate_devices(user_devices)
                assert all(
                    call[1]["condition"] is False
                    for call in mock_handle_error.call_args_list
                )


def test_validate_devices_with_invalid_device() -> None:
    user_devices = ["hip://0", "local-sync://default", "cuda://default"]
    user_drivers = {"hip", "local-sync", "cuda"}

    with patch(f"{libtuner.__name__}.extract_driver_names", return_value=user_drivers):
        with patch(
            f"{libtuner.__name__}.fetch_available_devices",
            return_value=["hip://0", "local-sync://default"],
        ):
            with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
                libtuner.validate_devices(user_devices)
                expected_call = call(
                    condition=True,
                    msg=f"Invalid device specified: cuda://default\nFetched available devices: ['hip://0', 'local-sync://default']",
                    error_type=argparse.ArgumentError,
                    exit_program=True,
                )
                assert expected_call in mock_handle_error.call_args_list


def test_get_compilation_success_rate():
    compiled_candidates = [0, None, 2, None, 4]
    assert libtuner.get_compilation_success_rate(compiled_candidates) == 3.0 / 5.0

    compiled_candidates = [0, 1, 2, 3, 4]
    assert libtuner.get_compilation_success_rate(compiled_candidates) == 1.0

    compiled_candidates = [None, None, None]
    assert libtuner.get_compilation_success_rate(compiled_candidates) == 0.0

    compiled_candidates = []
    assert libtuner.get_compilation_success_rate(compiled_candidates) == 0.0


def test_enum_collision():
    from iree.compiler.dialects import linalg, vector, iree_gpu, iree_codegen  # type: ignore


def test_baseline_result_handler_valid():
    handler = libtuner.BaselineResultHandler()
    assert not handler.is_valid()
    baseline = [
        libtuner.BenchmarkResult(0, 0.5, "hip://0"),
        libtuner.BenchmarkResult(0, math.inf, "hip://1"),
        libtuner.BenchmarkResult(0, 0.7, "hip://0"),
    ]
    assert libtuner.BaselineResultHandler.are_baseline_devices_unique([])
    assert not libtuner.BaselineResultHandler.are_baseline_devices_unique(baseline)
    handler.add_run(baseline)
    assert handler.is_valid()
    assert handler.is_valid_for_device("hip://0")
    assert not handler.is_valid_for_device("hip://1")

    assert handler.device_baseline_results["hip://0"] == [
        libtuner.BenchmarkResult(0, 0.5, "hip://0"),
        libtuner.BenchmarkResult(0, 0.7, "hip://0"),
    ]
    assert handler.device_baseline_results["hip://1"] == [
        libtuner.BenchmarkResult(0, math.inf, "hip://1"),
    ]

    assert handler.get_valid_time_ms("hip://0") == [0.5, 0.7]
    assert handler.get_valid_time_ms("hip://1") == []
    assert handler.get_valid_time_ms("hip://2") == []

    additional_baseline = [
        libtuner.BenchmarkResult(0, math.inf, "hip://1"),
        libtuner.BenchmarkResult(0, math.nan, "hip://1"),
        libtuner.BenchmarkResult(0, 1.2, "hip://1"),
        libtuner.BenchmarkResult(0, 0.8, "hip://1"),
    ]
    assert not libtuner.BaselineResultHandler.are_baseline_devices_unique(
        additional_baseline
    )
    handler.add_run(additional_baseline)
    assert handler.get_valid_time_ms("hip://0") == [0.5, 0.7]
    assert handler.get_valid_time_ms("hip://1") == [1.2, 0.8]
    assert handler.is_valid_for_device("hip://1")

    assert handler.get_average_result_ms("hip://0") == 0.6
    assert handler.get_average_result_ms("hip://1") == 1.0


def test_baseline_result_handler_speedup():
    handler = libtuner.BaselineResultHandler()
    first_run_baseline = [
        libtuner.BenchmarkResult(0, 1.0, "hip://0"),
        libtuner.BenchmarkResult(0, 0.5, "hip://1"),
    ]
    handler.detect_regressions(first_run_baseline) == []
    handler.add_run(first_run_baseline)

    second_run_baseline = [
        libtuner.BenchmarkResult(0, 0.8, "hip://0"),
        libtuner.BenchmarkResult(0, math.inf, "hip://1"),
        libtuner.BenchmarkResult(0, 1.2, "hip://2"),
    ]
    handler.detect_regressions(second_run_baseline) == ["hip:://1"]
    handler.add_run(second_run_baseline)

    candidates = [
        libtuner.BenchmarkResult(1, 0.4, "hip://0"),
        libtuner.BenchmarkResult(2, 0.3, "hip://1"),
        libtuner.BenchmarkResult(3, 1.0, "hip://2"),
        libtuner.BenchmarkResult(4, 0.2, "hip://3"),
    ]
    all_candidates_with_speedup = handler.get_candidates_ordered_by_speedup(candidates)
    assert all_candidates_with_speedup == [
        (candidates[3], 0.2 / 0.875),
        (candidates[0], 0.4 / 0.9),
        (candidates[1], 0.3 / 0.5),
        (candidates[2], 1.0 / 1.2),
    ]
    top_candidates_with_speedup = all_candidates_with_speedup[:2]
    assert [candidate.candidate_id for candidate, _ in top_candidates_with_speedup] == [
        4,
        1,
    ]

    candidates = [
        libtuner.BenchmarkResult(5, 0.6, "hip://0"),
        libtuner.BenchmarkResult(6, 0.4, "hip://1"),
        libtuner.BenchmarkResult(7, 0.8, "hip://2"),
    ]
    all_candidates_with_speedup = handler.get_candidates_ordered_by_speedup(candidates)
    assert all_candidates_with_speedup == [
        (candidates[0], 0.6 / 0.9),
        (candidates[2], 0.8 / 1.2),
        (candidates[1], 0.4 / 0.5),
    ]
    top_candidates_with_speedup = all_candidates_with_speedup[:2]
    assert [candidate.candidate_id for candidate, _ in top_candidates_with_speedup] == [
        5,
        7,
    ]

    handler = libtuner.BaselineResultHandler()
    all_candidates_with_speedup = handler.get_candidates_ordered_by_speedup(candidates)
    assert all_candidates_with_speedup == [
        (candidates[1], 1.0),
        (candidates[0], 1.0),
        (candidates[2], 1.0),
    ]
    top_candidates_with_speedup = all_candidates_with_speedup[:2]
    assert [candidate.candidate_id for candidate, _ in top_candidates_with_speedup] == [
        6,
        5,
    ]
