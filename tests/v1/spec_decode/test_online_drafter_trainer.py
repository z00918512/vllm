# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for EagleOnlineDrafterTrainer (CPU-only, no HF hub access).

These tests use a tiny self-contained mock model that mirrors the EAGLE3
interface (forward takes input_ids + hidden_states, returns an object with
.logits) but needs no pretrained weights.
"""

import torch
import torch.nn as nn

from vllm.v1.spec_decode.online_drafter_trainer import (
    DrafterTrainingConfig,
    EagleOnlineDrafterTrainer,
    _ReplayBuffer,
    _unfreeze_eagle_trainable_params,
)

# ---------------------------------------------------------------------------
# Minimal mock EAGLE3 model
# ---------------------------------------------------------------------------

_HIDDEN = 16
_AUX = _HIDDEN * 3  # 3 target layers concatenated
_VOCAB = 32


class _FakeOutput:
    """Mimics transformers CausalLMOutput just enough for the trainer."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _TinyEagleDraft(nn.Module):
    """Minimal EAGLE-like model.

    Parameter names deliberately contain the keywords 'fc' and 'layers' so
    that _unfreeze_eagle_trainable_params() unfreezes them, while embed_tokens
    and lm_head stay frozen — mirroring what happens in a real EAGLE3 model
    where those are shared with the target model.
    """

    def __init__(self) -> None:
        super().__init__()
        # Frozen (shared with target in a real EAGLE setup)
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)
        # Trainable — keyword 'fc' is matched by _unfreeze_eagle_trainable_params
        self.fc = nn.Linear(_AUX, _HIDDEN, bias=False)
        # Trainable — keyword 'layers' is matched
        self.layers = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN)])

    def forward(
        self,
        input_ids: torch.Tensor,  # [1, T]
        hidden_states: torch.Tensor,  # [1, T, _AUX]
    ) -> _FakeOutput:
        h = self.fc(hidden_states) + self.embed_tokens(input_ids)  # [1, T, H]
        for layer in self.layers:
            h = layer(h)
        return _FakeOutput(self.lm_head(h))  # logits: [1, T, V]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_trainer(
    num_steps_per_update: int = 2,
    update_interval_rl_steps: int = 1,
    replay_buffer_max_tokens: int = 512,
    **kw,
) -> EagleOnlineDrafterTrainer:
    model = _TinyEagleDraft()
    for p in model.parameters():
        p.requires_grad_(False)
    _unfreeze_eagle_trainable_params(model)

    cfg = DrafterTrainingConfig(
        num_steps_per_update=num_steps_per_update,
        update_interval_rl_steps=update_interval_rl_steps,
        replay_buffer_max_tokens=replay_buffer_max_tokens,
        **kw,
    )
    return EagleOnlineDrafterTrainer(model=model, config=cfg, device="cpu")


def _rand_aux(T: int) -> torch.Tensor:
    return torch.randn(T, _AUX)


def _rand_ids(T: int) -> torch.Tensor:
    return torch.randint(0, _VOCAB, (T,))


# ---------------------------------------------------------------------------
# _ReplayBuffer tests
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = _ReplayBuffer(max_tokens=100)
        aux = _rand_aux(10)
        ids = _rand_ids(10)
        buf.add(aux, ids)

        assert len(buf) == 10
        aux_out, ids_out, rew_out = buf.sample()
        assert aux_out.shape == (10, _AUX)
        assert ids_out.shape == (10,)
        assert rew_out.shape == (10,)
        assert torch.allclose(rew_out, torch.ones(10))

    def test_custom_reward_broadcast(self):
        buf = _ReplayBuffer(max_tokens=100)
        buf.add(_rand_aux(5), _rand_ids(5), reward=2.5)
        _, _, rew = buf.sample()
        assert torch.allclose(rew, torch.full((5,), 2.5))

    def test_eviction_on_overflow(self):
        buf = _ReplayBuffer(max_tokens=15)
        buf.add(_rand_aux(10), _rand_ids(10))
        buf.add(_rand_aux(10), _rand_ids(10))  # evicts first entry
        assert len(buf) == 10

    def test_clear(self):
        buf = _ReplayBuffer(max_tokens=100)
        buf.add(_rand_aux(10), _rand_ids(10))
        buf.clear()
        assert len(buf) == 0

    def test_multiple_seqs_concatenated(self):
        buf = _ReplayBuffer(max_tokens=200)
        for _ in range(3):
            buf.add(_rand_aux(10), _rand_ids(10))
        aux, ids, rew = buf.sample()
        assert aux.shape == (30, _AUX)
        assert ids.shape == (30,)
        assert rew.shape == (30,)


# ---------------------------------------------------------------------------
# EagleOnlineDrafterTrainer tests
# ---------------------------------------------------------------------------


class TestEagleOnlineDrafterTrainer:
    def test_should_update_requires_data_and_step(self):
        trainer = _make_trainer()
        # No data, no step.
        assert not trainer.should_update()

        trainer.add_rollout_data(_rand_aux(10), _rand_ids(10))
        # Data present but step not incremented.
        assert not trainer.should_update()

        trainer.increment_rl_step()
        assert trainer.should_update()

    def test_should_update_empty_buffer_after_clear(self):
        trainer = _make_trainer()
        trainer.add_rollout_data(_rand_aux(10), _rand_ids(10))
        trainer.increment_rl_step()
        trainer.train_step()  # clears buffer internally
        assert not trainer.should_update()

    def test_train_step_returns_positive_finite_loss(self):
        trainer = _make_trainer()
        trainer.add_rollout_data(_rand_aux(20), _rand_ids(20))
        trainer.increment_rl_step()

        loss = trainer.train_step()

        assert isinstance(loss, float)
        assert loss > 0.0
        assert loss == loss  # not NaN

    def test_train_step_clears_buffer(self):
        trainer = _make_trainer()
        trainer.add_rollout_data(_rand_aux(20), _rand_ids(20))
        trainer.increment_rl_step()
        trainer.train_step()
        assert len(trainer._buffer) == 0

    def test_reward_weighted_loss_runs(self):
        trainer = _make_trainer(reward_weight=1.0)
        trainer.add_rollout_data(_rand_aux(20), _rand_ids(20), reward=3.0)
        trainer.increment_rl_step()

        loss = trainer.train_step()

        assert isinstance(loss, float)
        assert loss > 0.0

    def test_get_trainable_state_dict_subset_and_cpu(self):
        trainer = _make_trainer()
        all_names = {n for n, _ in trainer.model.named_parameters()}
        sd = trainer.get_trainable_state_dict()

        assert set(sd.keys()).issubset(all_names)
        assert len(sd) < len(all_names)  # embed_tokens and lm_head are frozen

        for v in sd.values():
            assert v.device.type == "cpu"

    def test_trainable_params_are_fc_and_layers(self):
        trainer = _make_trainer()
        sd = trainer.get_trainable_state_dict()
        assert "fc.weight" in sd
        assert any(k.startswith("layers.") for k in sd)
        assert "embed_tokens.weight" not in sd
        assert "lm_head.weight" not in sd

    def test_too_short_buffer_returns_zero(self):
        trainer = _make_trainer()
        # Only 1 token — cannot shift to get a target.
        trainer.add_rollout_data(_rand_aux(1), _rand_ids(1))
        trainer.increment_rl_step()
        loss = trainer.train_step()
        assert loss == 0.0

    def test_update_interval_skips_steps(self):
        trainer = _make_trainer(update_interval_rl_steps=3, reward_weight=0.0)
        trainer.add_rollout_data(_rand_aux(10), _rand_ids(10))

        trainer.increment_rl_step()  # step 1
        assert not trainer.should_update()
        trainer.increment_rl_step()  # step 2
        assert not trainer.should_update()
        trainer.increment_rl_step()  # step 3 — interval hit
        assert trainer.should_update()

    def test_full_pipeline(self):
        """End-to-end: add data → check → train → export state dict."""
        trainer = _make_trainer()
        T = 30
        trainer.add_rollout_data(_rand_aux(T), _rand_ids(T))
        trainer.increment_rl_step()

        assert trainer.should_update()
        loss = trainer.train_step()
        assert loss > 0.0

        sd = trainer.get_trainable_state_dict()
        assert "fc.weight" in sd
        # Buffer cleared; no more update until new data arrives.
        assert not trainer.should_update()
