"""Sensor-patch masking: a fraction of the student's patch embeddings is replaced by a learnable [MASK]."""

import torch
import torch.nn as nn


def _valid_patch_mask_from_attn(sensor_attn_mask: list[list[torch.Tensor]]) -> torch.Tensor:
    """``(B, nvar, P)`` patch validity from SLIP's nested ``[batch][nvar]`` of ``(P, ps_i)`` masks."""
    per_sample = []
    for sample in sensor_attn_mask:  # [nvar] of (P, ps_i)
        per_ch = [ch.any(dim=-1) for ch in sample]  # each (P,)
        per_sample.append(torch.stack(per_ch, dim=0))  # (nvar, P)
    return torch.stack(per_sample, dim=0)  # (B, nvar, P)


def sample_patch_mask(sensor_attn_mask: list[list[torch.Tensor]], mask_ratio: float) -> torch.Tensor:
    """``(B, nvar, P)`` mask of ``round(mask_ratio * n_valid)`` random valid patches per (sample, channel)."""
    valid_patch_mask = _valid_patch_mask_from_attn(sensor_attn_mask)
    B, N, P = valid_patch_mask.shape
    noise = torch.rand(B, N, P, device=valid_patch_mask.device)
    noise = noise.masked_fill(~valid_patch_mask, float("inf"))
    ids_shuffle = noise.argsort(dim=-1)
    ranks = ids_shuffle.argsort(dim=-1)
    num_valid = valid_patch_mask.sum(dim=-1, keepdim=True)
    num_mask = (num_valid.float() * mask_ratio).round().long()
    return (ranks >= (num_valid - num_mask)) & (ranks < num_valid)


class SensorMaskApplier(nn.Module):
    """Holds the learnable ``[MASK]`` embedding and substitutes it at masked patch positions."""

    def __init__(self, hidden_dim: int, mask_ratio: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def sample_mask(self, sensor_attn_mask: list[list[torch.Tensor]]) -> torch.Tensor:
        """SLIP: exact-count ``(B, nvar, P)`` mask over the valid patches."""
        return sample_patch_mask(sensor_attn_mask, self.mask_ratio)

    def sample_mask_flamingo(self, sensor_attn_mask: torch.Tensor) -> torch.Tensor:
        """OpenTSLM-Flamingo: Bernoulli ``(B, C, N)`` mask over the valid patches of a dense validity tensor."""
        valid = sensor_attn_mask.bool()
        rand = torch.rand(valid.shape, device=valid.device)
        return (rand < self.mask_ratio) & valid

    def apply_post_patch_embed(self, patch_embed_out: torch.Tensor, bool_mask: torch.Tensor) -> torch.Tensor:
        """SLIP / Flamingo: replace the masked rows of a ``(B*nvar, P, D)`` patch embedding with ``[MASK]``."""
        B, nvar, P = bool_mask.shape
        BV, P_out, D = patch_embed_out.shape
        assert BV == B * nvar and P_out == P, (
            f"patch_embed output shape mismatch: got {tuple(patch_embed_out.shape)}, "
            f"expected ({B * nvar}, {P}, hidden_dim) from bool_mask {tuple(bool_mask.shape)}"
        )
        flat = bool_mask.reshape(B * nvar, P, 1).to(patch_embed_out.device)
        mask_token = self.mask_token.to(patch_embed_out.dtype).view(1, 1, D)
        return torch.where(flat, mask_token.expand_as(patch_embed_out), patch_embed_out)

    def apply_chatts_mask(self, patch_out: torch.Tensor) -> torch.Tensor:
        """ChatTS: Bernoulli(mask_ratio) over every leading position of a ``(..., D)`` patch embedding."""
        if patch_out.dim() < 2:
            raise ValueError(f"Expected a (..., D) patch embedding from ts_encoder.mlp, got {tuple(patch_out.shape)}.")
        D = patch_out.shape[-1]
        lead = patch_out.shape[:-1]  # (sum_patches,) or (B, nvar, N)
        bern = (torch.rand(lead, device=patch_out.device) < self.mask_ratio).unsqueeze(-1)
        mask_token = self.mask_token.to(patch_out.dtype).view(*([1] * len(lead)), D)
        return torch.where(bern, mask_token.expand_as(patch_out), patch_out)


def find_sensor_patch_embed(model: nn.Module) -> tuple[nn.Module, str, int]:
    """Return ``(module, backend, hidden_dim)`` of the per-patch embedding to mask."""
    cur = model

    # Unwrap only until the encoder is visible: ChatTS's `base_model` is its inner LM.
    def _has_target(m):
        return hasattr(m, "sensor_encoder") or hasattr(m, "ts_encoder")

    if not _has_target(cur) and getattr(cur, "module", None) is not None:
        cur = cur.module
    if not _has_target(cur) and getattr(cur, "base_model", None) is not None:
        cur = cur.base_model
    if not _has_target(cur) and hasattr(cur, "model"):
        cur = cur.model

    # Checked first: the Flamingo wrapper also exposes its CNN tokenizer as `sensor_encoder`.
    if getattr(getattr(cur, "config", None), "model_type", None) == "opentslm_flamingo" and hasattr(
        cur, "vision_encoder"
    ):
        return cur.vision_encoder, "flamingo", int(cur.embed_dim)

    if hasattr(cur, "sensor_encoder"):
        patch_embed = cur.sensor_encoder.patch_embed
        hidden_size = getattr(patch_embed, "hidden_size", None)
        if hidden_size is None:
            hidden_size = int(cur.sensor_encoder.embed_dim)
        return patch_embed, "slip", hidden_size

    if hasattr(cur, "ts_encoder"):
        ts_enc = cur.ts_encoder
        return ts_enc.mlp, "chatts", int(ts_enc.hidden_size)

    raise AttributeError(
        "Sensor masking needs a `sensor_encoder` (SLIP), `vision_encoder` (OpenTSLM-Flamingo) or "
        "`ts_encoder` (ChatTS) on the model."
    )
