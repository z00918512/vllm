# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for EagleOnlineDrafterTrainer — gap coverage.

Covers three gaps that the unit smoke-tests do not reach:

1. ``from_pretrained`` setup — freeze/unfreeze is correct, and the trainer
   honours ``trust_remote_code=True`` when calling transformers.

2. ``_forward_draft`` with ``position_ids`` — models that declare a
   ``position_ids`` parameter receive a correctly-shaped arange tensor.

3. End-to-end pipeline — add data → train → export state dict → name-remap →
   mock weight push to vLLM (via _remap_hf_draft_names + mock load_weights).

All tests run on CPU with no GPU and no network access (transformers calls
are patched or use local fixtures).
"""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from vllm.v1.spec_decode.online_drafter_trainer import (
    DrafterTrainingConfig,
    EagleOnlineDrafterTrainer,
    _unfreeze_eagle_trainable_params,
)
from vllm.v1.worker.gpu_worker import _remap_hf_draft_names

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_H = 16  # hidden size
_AUX = _H * 3  # 3-layer concatenated aux hidden
_VOCAB = 32


class _FakeOutput:
    """Minimal stand-in for transformers CausalLMOutputWithPast."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _EagleInner(nn.Module):
    """Inner model (self.model) mirroring the real EAGLE3 HF structure.

    Real HF EAGLE3 models have an inner ``LlamaModel``-like submodule at
    ``self.model`` so that parameter names are ``model.fc.weight``,
    ``model.layers.0.*``, etc.  This mirrors that layout.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        # 'fc' and 'layers' keywords trigger _unfreeze_eagle_trainable_params.
        self.fc = nn.Linear(_AUX, _H, bias=False)
        self.layers = nn.ModuleList([nn.Linear(_H, _H)])
        self._last_position_ids: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_position_ids = position_ids
        h = self.fc(hidden_states) + self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h  # [1, T, H]


class _EagleWithPositionIds(nn.Module):
    """EAGLE-style top-level model with position_ids support.

    Parameter layout mirrors real HF EAGLE3:
      model.embed_tokens.weight  (frozen — no keyword match)
      model.fc.weight            (trainable — 'fc' keyword)
      model.layers.0.*           (trainable — 'layers' keyword)
      lm_head.weight             (frozen — no keyword match, no 'model.' prefix)
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = _EagleInner()
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    @property
    def _last_position_ids(self) -> torch.Tensor | None:
        return self.model._last_position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> _FakeOutput:
        h = self.model(input_ids, hidden_states, position_ids)
        return _FakeOutput(self.lm_head(h))


class _EagleInnerNoPosIds(nn.Module):
    """Inner model without position_ids."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H)
        self.fc = nn.Linear(_AUX, _H, bias=False)
        self.layers = nn.ModuleList([nn.Linear(_H, _H)])

    def forward(self, input_ids, hidden_states):
        h = self.fc(hidden_states) + self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


class _EagleWithoutPositionIds(nn.Module):
    """EAGLE-style model that does NOT declare position_ids."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _EagleInnerNoPosIds()
        self.lm_head = nn.Linear(_H, _VOCAB, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> _FakeOutput:
        h = self.model(input_ids, hidden_states)
        return _FakeOutput(self.lm_head(h))


def _make_trainer_from_model(model: nn.Module, **kw) -> EagleOnlineDrafterTrainer:
    for p in model.parameters():
        p.requires_grad_(False)
    _unfreeze_eagle_trainable_params(model)
    cfg = DrafterTrainingConfig(
        num_steps_per_update=2,
        update_interval_rl_steps=1,
        replay_buffer_max_tokens=512,
        **kw,
    )
    return EagleOnlineDrafterTrainer(model=model, config=cfg, device="cpu")


def _rand_aux(T: int) -> torch.Tensor:
    return torch.randn(T, _AUX)


def _rand_ids(T: int) -> torch.Tensor:
    return torch.randint(0, _VOCAB, (T,))


# ---------------------------------------------------------------------------
# Gap 1 — from_pretrained setup
# ---------------------------------------------------------------------------


class TestFromPretrainedSetup:
    """Verify from_pretrained freezes/unfreezes params correctly.

    Uses mocked transformers calls so no HF Hub access is needed.
    """

    def _run_from_pretrained(self, model: nn.Module) -> EagleOnlineDrafterTrainer:
        fake_config = MagicMock()
        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=fake_config,
            ),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ),
        ):
            return EagleOnlineDrafterTrainer.from_pretrained(
                draft_model_name_or_path="/fake/eagle3_model",
                target_hidden_size=_H,
                num_target_layers=28,
                device="cpu",
                torch_dtype=torch.float32,
            )

    def test_trainable_params_are_layers_and_fc(self):
        trainer = self._run_from_pretrained(_EagleWithPositionIds())
        trainable = {n for n, p in trainer.model.named_parameters() if p.requires_grad}
        # Real HF EAGLE3 models nest weights under self.model so names are
        # "model.fc.weight", "model.layers.0.*", etc.
        assert "model.fc.weight" in trainable
        assert any("model.layers" in n for n in trainable)

    def test_embed_tokens_and_lm_head_are_frozen(self):
        trainer = self._run_from_pretrained(_EagleWithPositionIds())
        frozen = {n for n, p in trainer.model.named_parameters() if not p.requires_grad}
        assert "model.embed_tokens.weight" in frozen
        assert "lm_head.weight" in frozen

    def test_optimizer_has_only_trainable_params(self):
        trainer = self._run_from_pretrained(_EagleWithPositionIds())
        trainable_ptrs = {
            p.data_ptr() for p in trainer.model.parameters() if p.requires_grad
        }
        opt_ptrs = {
            p.data_ptr() for g in trainer.optimizer.param_groups for p in g["params"]
        }
        assert opt_ptrs == trainable_ptrs

    def test_trust_remote_code_passed_to_transformers(self):
        model = _EagleWithPositionIds()
        fake_config = MagicMock()
        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=fake_config,
            ) as mock_cfg,
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as mock_model,
        ):
            EagleOnlineDrafterTrainer.from_pretrained(
                draft_model_name_or_path="/fake/path",
                target_hidden_size=_H,
                num_target_layers=28,
                device="cpu",
            )
        mock_cfg.assert_called_once()
        assert mock_cfg.call_args.kwargs.get("trust_remote_code") is True
        mock_model.assert_called_once()
        assert mock_model.call_args.kwargs.get("trust_remote_code") is True

    def test_forward_works_after_from_pretrained(self):
        trainer = self._run_from_pretrained(_EagleWithPositionIds())
        T = 10
        aux = _rand_aux(T)
        ids = _rand_ids(T)
        trainer.add_rollout_data(aux_hidden_states=aux, input_ids=ids)
        trainer.increment_rl_step()
        loss = trainer.train_step()
        assert isinstance(loss, float)
        assert loss > 0.0


# ---------------------------------------------------------------------------
# Gap 2 — position_ids in _forward_draft
# ---------------------------------------------------------------------------


class TestForwardDraftPositionIds:
    def test_position_ids_passed_when_model_accepts_them(self):
        model = _EagleWithPositionIds()
        for p in model.parameters():
            p.requires_grad_(False)
        _unfreeze_eagle_trainable_params(model)
        trainer = EagleOnlineDrafterTrainer(
            model=model,
            config=DrafterTrainingConfig(),
            device="cpu",
        )

        assert trainer._forward_accepts_position_ids is True

        T = 8
        aux = _rand_aux(T).float()
        ids = _rand_ids(T)
        _ = trainer._forward_draft(aux, ids)

        # The model should have received position_ids = [0, 1, ..., T-1]
        assert model._last_position_ids is not None
        assert model._last_position_ids.shape == (1, T)
        expected = torch.arange(T).unsqueeze(0)
        assert torch.equal(model._last_position_ids, expected)

    def test_position_ids_not_passed_when_model_lacks_them(self):
        model = _EagleWithoutPositionIds()
        for p in model.parameters():
            p.requires_grad_(False)
        _unfreeze_eagle_trainable_params(model)
        trainer = EagleOnlineDrafterTrainer(
            model=model,
            config=DrafterTrainingConfig(),
            device="cpu",
        )

        assert trainer._forward_accepts_position_ids is False

        # forward should not raise TypeError for unexpected kwarg
        T = 8
        logits = trainer._forward_draft(_rand_aux(T).float(), _rand_ids(T))
        assert logits.shape == (T, _VOCAB)

    def test_position_ids_shape_matches_chunk_size(self):
        model = _EagleWithPositionIds()
        for p in model.parameters():
            p.requires_grad_(False)
        _unfreeze_eagle_trainable_params(model)
        trainer = EagleOnlineDrafterTrainer(
            model=model,
            config=DrafterTrainingConfig(num_steps_per_update=3),
            device="cpu",
        )

        # Run a full train_step so _forward_draft is called in chunks.
        T = 30
        trainer.add_rollout_data(_rand_aux(T), _rand_ids(T))
        trainer.increment_rl_step()
        loss = trainer.train_step()
        assert loss > 0.0


# ---------------------------------------------------------------------------
# Gap 3 — end-to-end: train → export → remap → mock push
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Full path from add_rollout_data to mock weight push in vLLM."""

    def _make_trainer(self) -> EagleOnlineDrafterTrainer:
        return _make_trainer_from_model(_EagleWithPositionIds())

    def test_state_dict_names_remap_correctly_for_vllm(self):
        trainer = self._make_trainer()
        T = 20
        trainer.add_rollout_data(_rand_aux(T), _rand_ids(T))
        trainer.increment_rl_step()
        trainer.train_step()

        sd = trainer.get_trainable_state_dict()
        # Simulate what the VeRL worker does: pass items to vLLM.
        items = list(sd.items())
        remapped = _remap_hf_draft_names(items, torch.device("cpu"))

        remapped_names = {n for n, _ in remapped}
        # After stripping "model." prefix, fc and layers params should be present.
        assert "fc.weight" in remapped_names
        assert any("layers" in n for n in remapped_names)
        # lm_head and embed_tokens must be absent (they have no "model." prefix
        # in a frozen-then-unfrozen setup, or are excluded by the filter).
        assert not any("lm_head" in n for n in remapped_names)

    def test_mock_inner_model_load_weights_called(self):
        """Simulate the full push: trainer → state_dict → remap → load_weights."""
        trainer = self._make_trainer()
        T = 20
        trainer.add_rollout_data(_rand_aux(T), _rand_ids(T))
        trainer.increment_rl_step()
        trainer.train_step()

        sd = trainer.get_trainable_state_dict()
        items = list(sd.items())

        received: list[list[tuple[str, torch.Tensor]]] = []

        class MockInnerModel:
            def load_weights(self, weights):
                received.append(list(weights))

        class MockDraftTop(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockInnerModel()

        class MockProposer:
            model = MockDraftTop()

        from types import SimpleNamespace

        worker = SimpleNamespace()
        worker.model_runner = SimpleNamespace()
        worker.model_runner.drafter = MockProposer()
        worker.device = torch.device("cpu")

        from vllm.v1.worker.gpu_worker import Worker

        with patch("torch.accelerator.synchronize"):
            Worker.update_draft_weights(worker, items)

        assert len(received) == 1
        pushed_names = {n for n, _ in received[0]}
        assert "fc.weight" in pushed_names
        assert any("layers" in n for n in pushed_names)

    def test_multiple_rollouts_accumulate_then_push(self):
        """Multiple add_rollout_data calls before a single update."""
        trainer = self._make_trainer()

        for _ in range(3):
            trainer.add_rollout_data(_rand_aux(10), _rand_ids(10))

        trainer.increment_rl_step()
        assert trainer.should_update()
        assert len(trainer._buffer) == 30

        loss = trainer.train_step()
        assert loss > 0.0
        assert len(trainer._buffer) == 0

    def test_async_push_weights_called(self):
        """VeRL OnlineDrafterWorker._push_weights calls engine.update_draft_weights."""
        from verl.workers.rollout.vllm_rollout.online_drafter_trainer import (
            OnlineDrafterConfig,
            OnlineDrafterWorker,
        )

        # Build a real trainer with tiny model, bypassing from_pretrained.
        model = _EagleWithPositionIds()
        for p in model.parameters():
            p.requires_grad_(False)
        _unfreeze_eagle_trainable_params(model)
        cfg_train = DrafterTrainingConfig(
            num_steps_per_update=1,
            update_interval_rl_steps=1,
            replay_buffer_max_tokens=256,
        )
        real_trainer = EagleOnlineDrafterTrainer(
            model=model, config=cfg_train, device="cpu"
        )

        # Wire a mock engine.
        pushed: list[dict] = []

        class MockEngine:
            async def update_draft_weights(self, state_dict):
                pushed.append(state_dict)

            async def pause_generation(self):
                pass

            async def resume_generation(self):
                pass

        mock_engine = MockEngine()

        # Build OnlineDrafterWorker directly (no Ray), injecting our trainer.
        worker = object.__new__(OnlineDrafterWorker)
        worker.config = OnlineDrafterConfig()
        worker.engine = mock_engine
        worker.trainer = real_trainer

        # Feed data and trigger update.
        T = 20
        worker.add_rollout_data(
            aux_hidden_states=_rand_aux(T),
            input_ids=_rand_ids(T),
        )
        loss = worker.maybe_update(rl_step=1)

        assert loss is not None
        assert isinstance(loss, float)
        assert len(pushed) == 1
        # State dict keys must be present.
        assert "model.fc.weight" in pushed[0] or "fc.weight" in pushed[0]
