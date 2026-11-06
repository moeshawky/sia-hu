"""Regression tests for D1: a non-2-D weight pair must be refused at the load
boundary with the offending shape named, and the effective-rank signal must not
leak a shape error out of its own boundary."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from model_headroom.models.loader import load_weight_pairs_from_json
from model_headroom.signals.effective_rank import compute_effective_rank


def _write_pairs(tmp_path: Any, payload: Any) -> str:
    path = tmp_path / "weight_pairs.json"
    path.write_text(json.dumps({"pairs": payload}))
    return str(path)


def test_loader_rejects_3d_fused_expert_tensor_naming_shape(tmp_path: Any) -> None:
    path = _write_pairs(
        tmp_path,
        [
            {
                "gradient": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                "weight": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                "layer_name": "layer_0",
            }
        ],
    )
    with pytest.raises(ValueError) as exc:
        load_weight_pairs_from_json(path)
    message = str(exc.value)
    assert "[2, 2, 2]" in message
    assert "layer_0" in message
    assert "2-D" in message


def test_loader_rejects_1d_vector_tensor_naming_shape(tmp_path: Any) -> None:
    path = _write_pairs(
        tmp_path,
        [
            {
                "gradient": [1.0, 2.0, 3.0, 4.0],
                "weight": [1.0, 2.0, 3.0, 4.0],
                "layer_name": "layer_1",
            }
        ],
    )
    with pytest.raises(ValueError) as exc:
        load_weight_pairs_from_json(path)
    message = str(exc.value)
    assert "[4]" in message
    assert "ndim 1" in message


def test_loader_rejects_0d_scalar_tensor_naming_shape(tmp_path: Any) -> None:
    path = _write_pairs(
        tmp_path,
        [
            {
                "gradient": 1.0,
                "weight": 2.0,
                "layer_name": "layer_scalar",
            }
        ],
    )
    with pytest.raises(ValueError) as exc:
        load_weight_pairs_from_json(path)
    message = str(exc.value)
    assert "ndim 0" in message
    assert "layer_scalar" in message


def test_loader_rejects_3d_tensor_with_zero_dimension(tmp_path: Any) -> None:
    path = _write_pairs(
        tmp_path,
        [
            {
                "gradient": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
                "weight": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                "layer_name": "layer_zero_dim",
            }
        ],
    )
    with pytest.raises(ValueError) as exc:
        load_weight_pairs_from_json(path)
    assert "[1, 2, 3]" in str(exc.value)


def test_loader_accepts_2d_pair_negative_control(tmp_path: Any) -> None:
    path = _write_pairs(
        tmp_path,
        [
            {
                "gradient": [[1.0, 2.0], [3.0, 4.0]],
                "weight": [[0.5, 0.5], [0.5, 0.5]],
                "layer_name": "layer_0",
            }
        ],
    )
    result = load_weight_pairs_from_json(path)
    assert result is not None
    assert len(result) == 1
    pair = result[0]
    assert pair.layer_name == "layer_0"
    assert tuple(pair.weight.shape) == (2, 2)
    assert tuple(pair.gradient.shape) == (2, 2)


def test_loader_accepts_empty_2d_pair_negative_control(tmp_path: Any) -> None:
    path = _write_pairs(tmp_path, [])
    assert load_weight_pairs_from_json(path) is None


def test_loader_missing_file_still_returns_none_negative_control() -> None:
    assert load_weight_pairs_from_json("/nonexistent/file.json") is None


def test_signal_direct_3d_call_returns_error_without_raising() -> None:
    tensor = torch.zeros(2, 2, 2, dtype=torch.float32)
    result = compute_effective_rank(tensor, sampling_mode="full_matrix")
    assert result.signal_status == "error"
    assert result.error_message is not None
    assert "[2, 2, 2]" in result.error_message
    assert "not 2-D" in result.error_message


def test_signal_direct_4d_call_returns_error_without_raising() -> None:
    tensor = torch.zeros(2, 3, 4, 5, dtype=torch.float32)
    result = compute_effective_rank(tensor)
    assert result.signal_status == "error"
    assert result.error_message is not None
    assert "[2, 3, 4, 5]" in result.error_message


def test_signal_direct_1d_call_is_degenerate_not_error() -> None:
    result = compute_effective_rank(torch.ones(64))
    assert result.signal_status == "degenerate"


def test_signal_direct_0d_call_is_degenerate_not_error() -> None:
    result = compute_effective_rank(torch.tensor(1.0))
    assert result.signal_status == "degenerate"


def test_signal_direct_empty_2d_call_does_not_crash() -> None:
    result = compute_effective_rank(torch.zeros(0, 0))
    assert result.signal_status in {"degenerate", "error"}


def test_signal_svd_failure_message_still_names_svd() -> None:
    result = compute_effective_rank(torch.eye(64))
    assert result.signal_status == "computed"
    assert result.error_message is None
