"""Patchify raw multivariate time series into the SLIP sensor-encoder input format."""

from typing import Optional, Union

import numpy as np
import torch


def get_patch_size(L: int) -> int:
    if L < 129:
        return 4
    if L < 513:
        return 16
    if L < 1025:
        return 32
    return 64


def _build_time_index(nvar: int, L: int) -> torch.Tensor:
    positions = torch.arange(1, L + 1, dtype=torch.float32)
    return positions.unsqueeze(0).expand(nvar, L) / (L + 1)


def raw_to_slip_cluster(
    raw_ts: Union[list, np.ndarray, torch.Tensor],
    patch_size: Optional[int] = None,
    normalize: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Patchify an ``(nvar, L)`` series into per-channel ``(sensors, masks, time_indices)``, left-padded."""
    if isinstance(raw_ts, str):
        raw_ts = [float(x) for x in raw_ts.split(",")]
    elif isinstance(raw_ts, (list, tuple)) and raw_ts and isinstance(raw_ts[0], str):
        raw_ts = [[float(x) for x in row.split(",")] for row in raw_ts]

    # Channels of unequal length are patchified one by one.
    if (
        isinstance(raw_ts, (list, tuple))
        and len(raw_ts) > 1
        and all(isinstance(ch, (list, tuple, np.ndarray)) for ch in raw_ts)
        and len({len(ch) for ch in raw_ts}) > 1
    ):
        sensors, masks, time_indices = [], [], []
        for ch in raw_ts:
            s, m, t = raw_to_slip_cluster(ch, patch_size=patch_size, normalize=normalize)
            sensors.extend(s)
            masks.extend(m)
            time_indices.extend(t)
        return sensors, masks, time_indices

    ts = np.asarray(raw_ts, dtype=np.float32)
    if ts.ndim == 1:
        ts = ts[None, :]
    if ts.ndim != 2:
        raise ValueError(f"raw_ts must be (nvar, L); got shape {ts.shape}")

    nvar, L = ts.shape
    sensor = torch.from_numpy(ts)
    mask = torch.ones_like(sensor, dtype=torch.bool) & ~torch.isnan(sensor)
    sensor = torch.nan_to_num(sensor, nan=0.0)

    if normalize:
        scale = sensor.abs().mean(dim=-1, keepdim=True) + 1e-6
        sensor = sensor / scale

    time_index = _build_time_index(nvar, L)

    if patch_size is None:
        patch_size = get_patch_size(L)

    remainder = L % patch_size
    if remainder != 0:
        pad_len = patch_size - remainder
        sensor = torch.nn.functional.pad(sensor, (pad_len, 0), mode="constant", value=0.0)
        mask = torch.nn.functional.pad(mask, (pad_len, 0), mode="constant", value=False)
        time_index = torch.nn.functional.pad(time_index, (pad_len, 0), mode="constant", value=0.0)

    num_patches = sensor.shape[-1] // patch_size
    sensor = sensor.reshape(nvar, num_patches, patch_size).contiguous()
    mask = mask.reshape(nvar, num_patches, patch_size).to(torch.float32).contiguous()
    time_index = time_index.reshape(nvar, num_patches, patch_size).contiguous()

    sensors = [sensor[i] for i in range(nvar)]
    masks = [mask[i] for i in range(nvar)]
    time_indices = [time_index[i] for i in range(nvar)]
    return sensors, masks, time_indices
