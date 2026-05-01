# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online drafter trainer for speculative decoding in RL pipelines.

Provides ``EagleOnlineDrafterTrainer``, a self-contained PyTorch training
wrapper for an EAGLE / EAGLE3 draft model.  It is designed to run in the RL
training process (e.g. inside a VeRL worker) rather than inside the vLLM
inference engine.

Typical usage (e.g. from a VeRL ``OnlineDrafterWorker``)::

    trainer = EagleOnlineDrafterTrainer.from_pretrained(
        draft_model_name_or_path="AngelSlim/Qwen3-1.7B_eagle3",
        target_hidden_size=2048,
        num_target_layers=28,
        device="cuda:0",
        config=DrafterTrainingConfig(lr=1e-4, num_steps_per_update=5),
    )

    # --- inside RL training loop ---
    # aux_hidden: [T, 3 * target_hidden_size]  (3 EAGLE3 layers concatenated)
    # input_ids:  [T]
    trainer.add_rollout_data(aux_hidden, input_ids)

    if trainer.should_update():
        loss = trainer.train_step()
        new_state_dict = trainer.get_trainable_state_dict()
        # push to vLLM:
        await vllm_engine.update_draft_weights(new_state_dict)

Training objective
------------------
Cross-entropy loss against the actual next tokens in the sequence::

    draft_logits = drafter(input_ids[:-1], aux_hidden[:-1])  # [T-1, V]
    loss = CE(draft_logits, input_ids[1:])

Optionally reward-weighted (ReSpec §3.2)::

    loss = (1 + reward_weight * r) * CE(...)
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DrafterTrainingConfig:
    """Hyperparameters for online drafter training."""

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0

    # Steps per update call (mini-batch gradient accumulation).
    num_steps_per_update: int = 5

    # Update every N RL steps (1 = every step, mirrors ReSpec Async-1).
    update_interval_rl_steps: int = 1

    # Max tokens to keep in the replay buffer.
    replay_buffer_max_tokens: int = 32_768

    # Reward weighting (set > 0 to enable reward-weighted loss as in ReSpec).
    reward_weight: float = 0.0

    # Gradient clipping norm.
    max_grad_norm: float = 1.0

    # dtype for training (default: match model dtype).
    dtype: torch.dtype | None = None


class _ReplayBuffer:
    """Simple ring buffer storing (aux_hidden, input_ids, reward)."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self._aux_hidden: list[torch.Tensor] = []
        self._input_ids: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._total_tokens: int = 0

    def add(
        self,
        aux_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        reward: float = 1.0,
    ) -> None:
        T = aux_hidden.shape[0]
        self._aux_hidden.append(aux_hidden.cpu())
        self._input_ids.append(input_ids.cpu())
        self._rewards.append(reward)
        self._total_tokens += T
        # Evict oldest entries if over capacity.
        while self._total_tokens > self.max_tokens and self._aux_hidden:
            evicted = self._aux_hidden.pop(0)
            self._input_ids.pop(0)
            self._rewards.pop(0)
            self._total_tokens -= evicted.shape[0]

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a concatenated batch of all buffered data."""
        aux = torch.cat(self._aux_hidden, dim=0)
        ids = torch.cat(self._input_ids, dim=0)
        T_per_seq = [t.shape[0] for t in self._aux_hidden]
        rewards = torch.tensor(
            [r for r, t in zip(self._rewards, T_per_seq) for _ in range(t)],
            dtype=torch.float32,
        )
        return aux, ids, rewards

    def clear(self) -> None:
        self._aux_hidden.clear()
        self._input_ids.clear()
        self._rewards.clear()
        self._total_tokens = 0

    def __len__(self) -> int:
        return self._total_tokens


class EagleOnlineDrafterTrainer:
    """Online trainer for an EAGLE / EAGLE3 draft model.

    Holds a trainable replica of the draft model, runs cross-entropy
    distillation against the actual next tokens collected during RL rollout,
    and exposes the updated state dict for hot-swapping into the vLLM
    inference engine via ``AsyncLLM.update_draft_weights()``.

    Parameters
    ----------
    model:
        The EAGLE draft model (``nn.Module``).  Embeddings and lm_head are
        expected to be tied to the target model and should be frozen externally
        before passing in.
    config:
        Training hyperparameters.
    device:
        Device to run training on.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DrafterTrainingConfig,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = config.dtype or next(model.parameters()).dtype

        self.model = model.to(self.device)

        # Only optimize non-frozen (non-shared) parameters.
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            logger.warning(
                "EagleOnlineDrafterTrainer: no trainable parameters found. "
                "Did you forget to unfreeze the draft-specific parameters?"
            )
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        self._buffer = _ReplayBuffer(config.replay_buffer_max_tokens)
        self._rl_step: int = 0

        # Probe the model's forward signature once so _forward_draft can pass
        # position_ids when the model supports it (standard transformers models
        # accept it; some lightweight HF EAGLE models may not).
        try:
            fwd_sig = inspect.signature(self.model.forward)
            self._forward_accepts_position_ids: bool = (
                "position_ids" in fwd_sig.parameters
            )
        except (TypeError, ValueError):
            self._forward_accepts_position_ids = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def increment_rl_step(self) -> None:
        """Call once per completed RL step to track the update cadence."""
        self._rl_step += 1

    def should_update(self) -> bool:
        """Return True when it is time to run a training pass."""
        return (
            self._rl_step > 0
            and self._rl_step % self.config.update_interval_rl_steps == 0
            and len(self._buffer) > 0
        )

    def add_rollout_data(
        self,
        aux_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        reward: float = 1.0,
    ) -> None:
        """Buffer training data collected from a single rollout sequence.

        Parameters
        ----------
        aux_hidden_states:
            Shape ``[T, 3 * target_hidden_size]``.  The concatenated hidden
            states from the three EAGLE3 target-model layers for each generated
            token.  Obtain by running the actor model with
            ``output_hidden_states=True`` and extracting layers
            [1, n//2-1, n-4].
        input_ids:
            Shape ``[T]``.  Token ids of the generated sequence.
        reward:
            Scalar reward for the trajectory (used for reward-weighted loss
            when ``config.reward_weight > 0``).
        """
        self._buffer.add(
            aux_hidden_states.detach().float(),
            input_ids.detach(),
            reward=reward,
        )

    def train_step(self) -> float:
        """Run one update pass over the buffered data.

        Returns the mean CE loss over all gradient steps.
        """
        aux, ids, rewards = self._buffer.sample()
        self._buffer.clear()

        if aux.shape[0] < 2:
            logger.warning("train_step: fewer than 2 tokens in buffer; skipping.")
            return 0.0

        aux = aux.to(self.device, dtype=self.dtype)
        ids = ids.to(self.device)
        rewards = rewards.to(self.device)

        # Shift: drafter at position t predicts token at t+1.
        aux_in = aux[:-1]  # [T-1, 3*h]  — context hidden states
        ids_in = ids[:-1]  # [T-1]        — context token ids
        ids_tgt = ids[1:]  # [T-1]        — next-token targets
        rew_in = rewards[:-1]  # [T-1]        — per-position reward weights

        T = aux_in.shape[0]
        chunk = max(1, T // self.config.num_steps_per_update)

        total_loss = 0.0
        self.model.train()
        for step in range(self.config.num_steps_per_update):
            lo = step * chunk
            hi = min(lo + chunk, T)
            if lo >= T:
                break

            draft_logits = self._forward_draft(aux_in[lo:hi], ids_in[lo:hi])
            loss = self._compute_loss(draft_logits, ids_tgt[lo:hi], rew_in[lo:hi])

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            total_loss += loss.item()

        self.model.eval()
        mean_loss = total_loss / self.config.num_steps_per_update
        logger.debug("drafter train_step loss=%.4f  T=%d", mean_loss, T)
        return mean_loss

    def get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a CPU state dict of trainable (non-shared) parameters only."""
        return {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_draft(
        self, aux_hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Run a training-compatible forward pass of the HF EAGLE draft model.

        Parameters
        ----------
        aux_hidden:
            Shape ``[T, 3 * target_hidden_size]``.  Concatenated EAGLE3 aux
            hidden states for positions 0..T-1.
        input_ids:
            Shape ``[T]``.  Token ids for positions 0..T-1.

        Returns
        -------
        torch.Tensor
            Shape ``[T, vocab_size]`` — unnormalised logits.
        """
        T = input_ids.shape[0]
        # HF EAGLE models expect a batch dimension.
        kwargs: dict = {
            "input_ids": input_ids.unsqueeze(0),  # [1, T]
            "hidden_states": aux_hidden.unsqueeze(0),  # [1, T, 3*h]
        }
        # Standard transformers models support position_ids for correct RoPE.
        # Pass them explicitly so teacher-forcing chunks get correct positions.
        if self._forward_accepts_position_ids:
            kwargs["position_ids"] = torch.arange(T, device=input_ids.device).unsqueeze(
                0
            )  # [1, T]

        output = self.model(**kwargs)

        # HF models return CausalLMOutput(WithPast) with a .logits attribute.
        if hasattr(output, "logits") and output.logits is not None:
            return output.logits.squeeze(0)  # [T, vocab_size]

        # Fallback: model returned raw hidden states (e.g. a custom EAGLE impl).
        # Apply lm_head manually.
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.squeeze(0)
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(hidden)
        raise RuntimeError(
            "Cannot extract logits from draft model output; "
            "model must expose either .logits or a .lm_head."
        )

    def _compute_loss(
        self,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss, optionally reward-weighted."""
        # draft_logits: [T, vocab_size], target_ids: [T]
        loss = F.cross_entropy(
            draft_logits.float(), target_ids, reduction="none"
        )  # [T]

        if self.config.reward_weight > 0:
            # Upweight positions from high-reward trajectories (ReSpec §3.2).
            w = 1.0 + self.config.reward_weight * rewards
            loss = loss * w

        return loss.mean()

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        draft_model_name_or_path: str,
        target_hidden_size: int,
        num_target_layers: int,
        device: torch.device | str = "cuda",
        config: DrafterTrainingConfig | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> EagleOnlineDrafterTrainer:
        """Load an EAGLE3 draft model from HuggingFace and wrap it for training.

        The embeddings and lm_head are frozen (they are shared with the target
        model during inference and should not diverge).  Only the EAGLE decoder
        layer(s) and the ``fc`` / ``combine_hidden_states`` projection are made
        trainable.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = config or DrafterTrainingConfig()

        hf_config = AutoConfig.from_pretrained(
            draft_model_name_or_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            draft_model_name_or_path,
            config=hf_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad_(False)

        # Unfreeze only the EAGLE-specific trainable parts.
        # For EAGLE3: the decoder layers + fc / combine_hidden_states.
        _unfreeze_eagle_trainable_params(model)

        return cls(model, config=cfg, device=device)


def _unfreeze_eagle_trainable_params(model: nn.Module) -> None:
    """Unfreeze the EAGLE-specific parameters (decoder layers + projection)."""
    trainable_submodule_keywords = (
        "layers",  # EAGLE decoder layer(s)
        "fc",  # fc / combine_hidden_states linear projection
        "combine_hidden",  # alternate naming
        "hidden_norm",  # normalisation inside EAGLE layer
    )
    unfrozen = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in trainable_submodule_keywords):
            param.requires_grad_(True)
            unfrozen += 1
    if unfrozen == 0:
        logger.warning(
            "_unfreeze_eagle_trainable_params: no parameters matched; "
            "all parameters remain frozen."
        )
    else:
        logger.info("Unfroze %d EAGLE draft parameters for online training.", unfrozen)
