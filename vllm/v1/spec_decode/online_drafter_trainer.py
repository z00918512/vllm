# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online drafter trainer for speculative decoding in RL pipelines.

Provides ``EagleOnlineDrafterTrainer``, a self-contained PyTorch training
wrapper for an EAGLE3 draft model.  It runs in the RL training process
(e.g. inside a VeRL worker) rather than inside the vLLM inference engine.

Architecture
------------
The draft model mirrors the AngelSlim / vLLM ``Eagle3LlamaForCausalLM``
layout (single midlayer + fc + final RMSNorm + draft lm_head + d2t/t2d
vocab maps).  The midlayer's qkv input dim is ``2 * hidden_size`` because
attention consumes the concatenation of ``[token_embeds, aux_hidden_post_fc]``.
EAGLE3 reuses the *target* model's embedding table — it does not own one —
so the trainer needs the path to the target model to clone a frozen copy.

Typical usage from a VeRL ``OnlineDrafterWorker``::

    trainer = EagleOnlineDrafterTrainer.from_pretrained(
        draft_model_name_or_path="AngelSlim/Qwen3-1.7B_eagle3",
        target_model_name_or_path="Qwen/Qwen3-1.7B",
        device="cuda:0",
        config=DrafterTrainingConfig(lr=1e-4, num_steps_per_update=5),
    )

    trainer.add_rollout_data(aux_hidden, input_ids)
    if trainer.should_update():
        loss = trainer.train_step()
        new_state_dict = trainer.get_trainable_state_dict()
        await vllm_engine.update_draft_weights(new_state_dict)

Training objective
------------------
Cross-entropy against the actual next tokens, in *draft* vocab space::

    draft_logits = drafter(input_ids[:-1], aux_hidden[:-1])  # [T-1, V_draft]
    targets_draft = t2d_remap(input_ids[1:])
    loss = CE(draft_logits, targets_draft)

Optionally reward-weighted (ReSpec §3.2)::

    loss = (1 + reward_weight * r) * CE(...)
"""

from __future__ import annotations

import logging
import os
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


# ---------------------------------------------------------------------------
# EAGLE3 draft model — pure-PyTorch port of vllm.model_executor.models.llama_eagle3
# ---------------------------------------------------------------------------


class Eagle3Attention(nn.Module):
    """Llama-style GQA attention with ``2 * hidden_size`` qkv input.

    EAGLE3's first (and only) midlayer consumes ``[embeds, hidden_states]``
    concatenated along the feature dim, so the q/k/v projections take
    ``2 * H`` inputs while the output projection produces ``H``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.attention_bias = getattr(config, "attention_bias", False)

        qkv_in = 2 * self.hidden_size
        self.q_proj = nn.Linear(qkv_in, self.q_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(qkv_in, self.kv_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(qkv_in, self.kv_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=self.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, 2H]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        # to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.q_dim)
        return self.o_proj(attn)


class Eagle3MidLayer(nn.Module):
    """Single EAGLE3 decoder layer (matches AngelSlim ``midlayer.*``).

    Forward semantics mirror vllm.model_executor.models.llama_eagle3
    .LlamaDecoderLayer for ``layer_idx == 0`` with the default
    ``norm_after_residual`` flow (residual taken pre-hidden_norm).
    """

    def __init__(self, config) -> None:
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(config)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        embeds: torch.Tensor,  # [B, T, H]
        hidden_states: torch.Tensor,  # [B, T, H]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        embeds_n = self.input_layernorm(embeds)
        residual = hidden_states
        hidden_n = self.hidden_norm(hidden_states)
        x = torch.cat([embeds_n, hidden_n], dim=-1)  # [B, T, 2H]
        x = self.self_attn(x, position_embeddings)
        x = x + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return x + residual


class Eagle3DraftModel(nn.Module):
    """Training-friendly EAGLE3 draft model.

    Layout (matches AngelSlim/Qwen3-1.7B_eagle3 checkpoint keys 1:1):
      fc.weight                              [H,    3 * target_H]
      midlayer.input_layernorm.weight        [H]
      midlayer.hidden_norm.weight            [H]
      midlayer.self_attn.{q,k,v,o}_proj.weight
      midlayer.post_attention_layernorm.weight [H]
      midlayer.mlp.{gate,up,down}_proj.weight
      norm.weight                            [H]
      lm_head.weight                         [draft_vocab_size, H]

    Buffers (loaded from checkpoint; not trainable):
      d2t  [draft_vocab_size]    draft-id offset to recover target id
      t2d  [vocab_size]          mask of which target ids are in draft vocab

    The ``embed_tokens`` table is a *frozen* clone of the target model's
    embedding — set via ``set_target_embed_tokens()`` because EAGLE3 reuses
    the target's embeds rather than learning its own.
    """

    def __init__(self, config) -> None:
        super().__init__()
        from transformers.models.llama.modeling_llama import (
            LlamaRMSNorm,
            LlamaRotaryEmbedding,
        )

        self.config = config
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)

        self.fc = nn.Linear(3 * target_hidden_size, self.hidden_size, bias=False)
        self.midlayer = Eagle3MidLayer(config)
        self.norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)

        self.register_buffer(
            "d2t",
            torch.zeros(self.draft_vocab_size, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "t2d",
            torch.zeros(config.vocab_size, dtype=torch.bool),
            persistent=True,
        )

        self.embed_tokens: nn.Embedding | None = None
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_target_embed_tokens(self, embed_tokens: nn.Embedding) -> None:
        """Attach a *frozen* copy of the target model's embedding table."""
        for p in embed_tokens.parameters():
            p.requires_grad_(False)
        self.embed_tokens = embed_tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        hidden_states: torch.Tensor,  # [B, T, 3 * target_H]
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError(
                "Eagle3DraftModel.set_target_embed_tokens() must be called "
                "before forward()."
            )

        h = self.fc(hidden_states)  # [B, T, H]
        embeds = self.embed_tokens(input_ids)  # [B, T, H]

        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[-1], device=input_ids.device
            ).unsqueeze(0)

        cos, sin = self.rotary_emb(h, position_ids)
        h = self.midlayer(
            embeds=embeds, hidden_states=h, position_embeddings=(cos, sin)
        )
        h = self.norm(h)
        return self.lm_head(h)  # [B, T, draft_vocab_size]

    # ------------------------------------------------------------------
    # Checkpoint loader (AngelSlim format)
    # ------------------------------------------------------------------

    def load_eagle3_state_dict(
        self, state_dict: dict[str, torch.Tensor], *, strict_warn: bool = True
    ) -> None:
        """Load an AngelSlim-format checkpoint dict (keys preserved 1:1)."""
        remapped: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k == "d2t":
                remapped["d2t"] = v.to(torch.long)
            elif k == "t2d":
                # Some checkpoints store t2d as float/int — coerce to bool.
                remapped["t2d"] = v.to(torch.bool)
            elif k.startswith("midlayer.") or k in (
                "fc.weight",
                "norm.weight",
                "lm_head.weight",
            ):
                remapped[k] = v
            elif strict_warn:
                logger.warning(
                    "Eagle3DraftModel: ignoring unknown checkpoint key %s", k
                )

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        # ``embed_tokens`` and ``rotary_emb.*`` are expected-missing.
        leftover_missing = [
            m
            for m in missing
            if not m.startswith("embed_tokens") and not m.startswith("rotary_emb")
        ]
        if leftover_missing:
            logger.warning(
                "Eagle3DraftModel: missing keys after load: %s", leftover_missing
            )
        if unexpected:
            logger.warning(
                "Eagle3DraftModel: unexpected keys after load: %s", unexpected
            )

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> Eagle3DraftModel:
        """Build an Eagle3DraftModel and load its AngelSlim-format checkpoint."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if not hasattr(config, "target_hidden_size"):
            config.target_hidden_size = config.hidden_size

        model = cls(config).to(dtype=torch_dtype)
        sd = _load_checkpoint_state_dict(path)
        model.load_eagle3_state_dict(sd)
        return model


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


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
        while self._total_tokens > self.max_tokens and self._aux_hidden:
            evicted = self._aux_hidden.pop(0)
            self._input_ids.pop(0)
            self._rewards.pop(0)
            self._total_tokens -= evicted.shape[0]

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class EagleOnlineDrafterTrainer:
    """Online trainer for an EAGLE3 draft model.

    Holds a trainable replica of the draft model, runs cross-entropy
    distillation against the actual next tokens collected during RL rollout,
    and exposes the updated state dict for hot-swapping into the vLLM
    inference engine via ``AsyncLLM.update_draft_weights()``.
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

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            logger.warning(
                "EagleOnlineDrafterTrainer: no trainable parameters. "
                "Did you forget to unfreeze the EAGLE-specific parts?"
            )
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        self._buffer = _ReplayBuffer(config.replay_buffer_max_tokens)
        self._rl_step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def increment_rl_step(self) -> None:
        self._rl_step += 1

    def should_update(self) -> bool:
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
        """Buffer training data from one rollout sequence.

        ``aux_hidden_states`` shape: ``[T, 3 * target_hidden_size]``.
        ``input_ids`` shape: ``[T]`` — *target* vocab ids.
        """
        self._buffer.add(
            aux_hidden_states.detach().float(),
            input_ids.detach(),
            reward=reward,
        )

    def train_step(self) -> float:
        aux, ids, rewards = self._buffer.sample()
        self._buffer.clear()

        if aux.shape[0] < 2:
            logger.warning("train_step: fewer than 2 tokens in buffer; skipping.")
            return 0.0

        aux = aux.to(self.device, dtype=self.dtype)
        ids = ids.to(self.device)
        rewards = rewards.to(self.device)

        aux_in = aux[:-1]
        ids_in = ids[:-1]
        ids_tgt = ids[1:]
        rew_in = rewards[:-1]

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
        """Return a CPU state dict in AngelSlim key naming.

        Keys match the original checkpoint layout (``fc.weight``,
        ``midlayer.*``, ``norm.weight``, ``lm_head.weight``) so they round-trip
        cleanly through ``Eagle3LlamaForCausalLM.load_weights`` on the vLLM
        side without any external remap.
        """
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
        """Forward through the EAGLE3 draft model.

        Returns logits of shape ``[T, draft_vocab_size]``.
        """
        T = input_ids.shape[0]
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        logits = self.model(
            input_ids=input_ids.unsqueeze(0),
            hidden_states=aux_hidden.unsqueeze(0),
            position_ids=position_ids,
        )
        return logits.squeeze(0)  # [T, draft_vocab_size]

    def _compute_loss(
        self,
        draft_logits: torch.Tensor,  # [T, draft_vocab_size]
        target_ids: torch.Tensor,  # [T]   — target-vocab ids
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """CE loss in *draft* vocab space, with ids remapped via d2t."""
        # Map target-vocab ids → draft-vocab ids via the inverse of d2t,
        # where d2t[draft_idx] = target_idx - draft_idx.
        ids_draft = _target_ids_to_draft_ids(self.model, target_ids)

        # Positions whose target token is *not* in the draft vocab get masked
        # out (they cannot be predicted by the draft head and would explode CE).
        valid = ids_draft >= 0
        if not valid.any():
            return draft_logits.new_zeros((), requires_grad=True)

        loss = F.cross_entropy(
            draft_logits[valid].float(),
            ids_draft[valid],
            reduction="none",
        )

        if self.config.reward_weight > 0:
            w = 1.0 + self.config.reward_weight * rewards[valid]
            loss = loss * w

        return loss.mean()

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        draft_model_name_or_path: str,
        target_model_name_or_path: str,
        device: torch.device | str = "cuda",
        config: DrafterTrainingConfig | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> EagleOnlineDrafterTrainer:
        """Load an EAGLE3 draft model + clone target embeddings, wrap for training.

        - ``draft_model_name_or_path``: HF path to the AngelSlim EAGLE3 checkpoint.
        - ``target_model_name_or_path``: HF path to the *target* model whose
          embedding table EAGLE3 reuses (frozen).
        """
        cfg = config or DrafterTrainingConfig()

        model = Eagle3DraftModel.from_pretrained(
            draft_model_name_or_path, torch_dtype=torch_dtype
        )

        # Pull a frozen copy of the target's embed_tokens.
        embed = _clone_target_embed_tokens(target_model_name_or_path, dtype=torch_dtype)
        model.set_target_embed_tokens(embed)

        # Freeze everything, then unfreeze EAGLE-specific trainable parts.
        for p in model.parameters():
            p.requires_grad_(False)
        _unfreeze_eagle_trainable_params(model)

        return cls(model, config=cfg, device=device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_checkpoint_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load ``pytorch_model.bin`` or ``model.safetensors`` from ``path``."""
    bin_path = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=False)

    st_path = os.path.join(path, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file

        return load_file(st_path)

    raise FileNotFoundError(
        f"Neither pytorch_model.bin nor model.safetensors found in {path}"
    )


def _clone_target_embed_tokens(
    target_model_path: str, dtype: torch.dtype
) -> nn.Embedding:
    """Return a standalone, frozen ``nn.Embedding`` cloned from the target model."""
    from transformers import AutoModelForCausalLM

    target = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    src = target.get_input_embeddings()
    cloned = nn.Embedding(
        src.num_embeddings,
        src.embedding_dim,
        padding_idx=src.padding_idx,
    ).to(dtype=dtype)
    with torch.no_grad():
        cloned.weight.copy_(src.weight)
    cloned.weight.requires_grad_(False)
    del target
    return cloned


def _target_ids_to_draft_ids(
    model: nn.Module, target_ids: torch.Tensor
) -> torch.Tensor:
    """Map target-vocab ids → draft-vocab ids via the model's ``d2t`` buffer.

    ``d2t[draft_id] = target_id - draft_id`` (the offset that recovers the
    target id from a draft id), so the inverse map can be built once and
    cached.  Positions whose target token is not in the draft vocab return
    ``-1``.
    """
    d2t: torch.Tensor = model.d2t  # [draft_vocab_size]
    vocab_size = model.config.vocab_size
    device = target_ids.device

    if not hasattr(model, "_target_to_draft_idx") or model._target_to_draft_idx is None:
        draft_ids = torch.arange(d2t.shape[0], device=d2t.device)
        target_for_draft = draft_ids + d2t  # [draft_vocab_size]
        idx = torch.full((vocab_size,), -1, dtype=torch.long, device=d2t.device)
        # Guard against out-of-range entries.
        valid = (target_for_draft >= 0) & (target_for_draft < vocab_size)
        idx[target_for_draft[valid]] = draft_ids[valid]
        model._target_to_draft_idx = idx

    return model._target_to_draft_idx.to(device)[target_ids]


def _unfreeze_eagle_trainable_params(model: nn.Module) -> None:
    """Unfreeze all EAGLE-specific parameters (everything except embed_tokens)."""
    unfrozen = 0
    for name, param in model.named_parameters():
        if name.startswith("embed_tokens"):
            continue
        param.requires_grad_(True)
        unfrozen += 1
    if unfrozen == 0:
        logger.warning("_unfreeze_eagle_trainable_params: no parameters matched.")
    else:
        logger.info("Unfroze %d EAGLE draft parameters for online training.", unfrozen)
