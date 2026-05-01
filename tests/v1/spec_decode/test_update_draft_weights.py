# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the draft-weight update path in gpu_worker.

These tests do NOT require a GPU or a live vLLM engine.  They verify that:
  1. _remap_hf_draft_names strips the "model." prefix and drops shared weights.
  2. update_draft_weights delegates to the inner LlamaModel.load_weights with
     correctly remapped names, so that QKV / gate-up shards are stacked.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm.v1.worker.gpu_worker import _remap_hf_draft_names

# ---------------------------------------------------------------------------
# Tests for _remap_hf_draft_names
# ---------------------------------------------------------------------------


class TestRemapHfDraftNames:
    def test_strips_model_prefix(self):
        items = [
            ("model.layers.0.self_attn.q_proj.weight", torch.zeros(4, 4)),
            ("model.fc.weight", torch.zeros(4, 4)),
        ]
        result = _remap_hf_draft_names(items, torch.device("cpu"))
        names = [n for n, _ in result]
        assert names == ["layers.0.self_attn.q_proj.weight", "fc.weight"]

    def test_drops_lm_head(self):
        items = [
            ("model.fc.weight", torch.zeros(4, 4)),
            ("lm_head.weight", torch.zeros(4, 4)),
        ]
        result = _remap_hf_draft_names(items, torch.device("cpu"))
        names = [n for n, _ in result]
        assert "lm_head.weight" not in names
        assert "fc.weight" in names

    def test_drops_embed_tokens_without_prefix(self):
        # If embed_tokens appears at top level (no model. prefix) it is
        # target-shared and must be dropped.
        items = [
            ("embed_tokens.weight", torch.zeros(4, 4)),
            ("model.fc.weight", torch.zeros(4, 4)),
        ]
        result = _remap_hf_draft_names(items, torch.device("cpu"))
        names = [n for n, _ in result]
        assert "embed_tokens.weight" not in names
        assert "fc.weight" in names

    def test_all_hf_fused_names_survive(self):
        hf_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.fc.weight",
        ]
        items = [(n, torch.zeros(2, 2)) for n in hf_names]
        result = _remap_hf_draft_names(items, torch.device("cpu"))
        expected = [n[len("model.") :] for n in hf_names]
        assert [n for n, _ in result] == expected

    def test_tensors_moved_to_device(self):
        items = [("model.fc.weight", torch.zeros(2, 2))]
        result = _remap_hf_draft_names(items, torch.device("cpu"))
        assert result[0][1].device.type == "cpu"

    def test_midlayer_prefix_passes_through(self):
        # midlayer. → layers.0. is handled by LlamaModel.load_weights, not here.
        items = [("model.midlayer.self_attn.q_proj.weight", torch.zeros(2, 2))]
        result = _remap_hf_draft_names(items, torch.device("cpu"))
        # Should be passed through as "midlayer.self_attn.q_proj.weight";
        # LlamaModel.load_weights will do the midlayer→layers.0 rename.
        assert result[0][0] == "midlayer.self_attn.q_proj.weight"

    def test_empty_input(self):
        assert _remap_hf_draft_names([], torch.device("cpu")) == []


# ---------------------------------------------------------------------------
# Tests for update_draft_weights delegation
# ---------------------------------------------------------------------------


class TestUpdateDraftWeightsDelegation:
    """Verify that update_draft_weights calls inner_model.load_weights with
    remapped names, without needing a GPU."""

    def _make_mock_worker(self):
        """Build a minimal Worker-like object with mocked internals."""
        from types import SimpleNamespace

        load_calls: list[list[tuple[str, torch.Tensor]]] = []

        class MockInnerModel:
            def load_weights(self, weights):
                load_calls.append(list(weights))

        class MockDraftTop(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockInnerModel()

        class MockProposer:
            model = MockDraftTop()

        class MockModelRunner:
            drafter = MockProposer()
            model = nn.Linear(2, 2)  # stand-in for target model

        worker = SimpleNamespace()
        worker.model_runner = MockModelRunner()
        worker.device = torch.device("cpu")
        return worker, load_calls

    def test_delegates_to_inner_load_weights(self):
        from vllm.v1.worker.gpu_worker import Worker  # noqa: E501

        worker, load_calls = self._make_mock_worker()

        items = [
            ("model.layers.0.self_attn.q_proj.weight", torch.zeros(4, 4)),
            ("model.fc.weight", torch.zeros(4, 4)),
        ]

        # Bind and call the method directly on our fake worker.
        with patch("torch.accelerator.synchronize"):
            Worker.update_draft_weights(worker, items)

        assert len(load_calls) == 1
        names = [n for n, _ in load_calls[0]]
        assert "layers.0.self_attn.q_proj.weight" in names
        assert "fc.weight" in names
        # lm_head must not appear
        assert not any("lm_head" in n for n in names)

    def test_lm_head_not_forwarded(self):
        from vllm.v1.worker.gpu_worker import Worker  # noqa: E501

        worker, load_calls = self._make_mock_worker()

        items = [
            ("model.fc.weight", torch.zeros(4, 4)),
            ("lm_head.weight", torch.zeros(4, 4)),
        ]

        with patch("torch.accelerator.synchronize"):
            Worker.update_draft_weights(worker, items)

        names = [n for n, _ in load_calls[0]] if load_calls else []
        assert not any("lm_head" in n for n in names)

    def test_no_items_skips_load_weights(self):
        from vllm.v1.worker.gpu_worker import Worker  # noqa: E501

        worker, load_calls = self._make_mock_worker()

        # Only non-model-prefixed items → all dropped → load_weights never called.
        items = [("lm_head.weight", torch.zeros(4, 4))]

        with patch("torch.accelerator.synchronize"):
            Worker.update_draft_weights(worker, items)

        assert load_calls == []

    def test_missing_drafter_raises(self):
        from types import SimpleNamespace

        from vllm.v1.worker.gpu_worker import Worker  # noqa: E501

        worker = SimpleNamespace()
        worker.model_runner = SimpleNamespace()
        worker.model_runner.drafter = None
        worker.device = torch.device("cpu")

        with (
            pytest.raises(RuntimeError, match="No draft model loaded"),
            patch("torch.accelerator.synchronize"),
        ):
            Worker.update_draft_weights(worker, [])

    def test_drafter_without_inner_model_raises(self):
        from types import SimpleNamespace

        from vllm.v1.worker.gpu_worker import Worker  # noqa: E501

        class DrafterNoInner:
            model = nn.Linear(2, 2)  # no .model.model attribute

        worker = SimpleNamespace()
        worker.model_runner = SimpleNamespace()
        worker.model_runner.drafter = DrafterNoInner()
        worker.device = torch.device("cpu")

        with (
            pytest.raises(RuntimeError, match="inner '.model'"),
            patch("torch.accelerator.synchronize"),
        ):
            Worker.update_draft_weights(
                worker,
                [("model.fc.weight", torch.zeros(2, 2))],
            )
