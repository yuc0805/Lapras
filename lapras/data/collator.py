# Copyright 2025 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER
from ..extras.packages import is_pillow_available
from .mm_plugin import FlamingoPlugin, SlipPlugin


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""Expand 2d attention mask to 4d attention mask.

    Expand the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    handle packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    # Create a non-padding mask.
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    # Create indices for comparison.
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    # Create a lower triangular mask.
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images, videos and audios.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            self.get_rope_func = self.model.get_rope_index  # transformers < 4.52.0 or qwen2.5 omni
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index  # transformers >= 4.52.0
        else:
            self.get_rope_func = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_audios, batch_timeseries = [], [], [], []
        batch_ts_patch_sizes = []
        batch_imglens, batch_vidlens, batch_audlens, batch_tslens, batch_input_ids = [], [], [], [], []
        # SLIP and Flamingo take one multivariate item per sample; ChatTS takes one item per channel.
        is_per_sample_ts = isinstance(self.template.mm_plugin, (SlipPlugin, FlamingoPlugin))
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            timeseries = feature.pop("timeseries", None) or []
            ts_patch_sizes = feature.pop("ts_patch_sizes", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            if is_per_sample_ts:
                if timeseries:
                    batch_timeseries.append(timeseries)
                    batch_ts_patch_sizes.append(ts_patch_sizes[0] if ts_patch_sizes else None)
            else:
                batch_timeseries.extend(timeseries)
                batch_ts_patch_sizes.extend(ts_patch_sizes)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_tslens.append(len(timeseries))
            batch_input_ids.append(feature["input_ids"])

        fake_input_ids = []
        if (
            self.template.mm_plugin.image_token is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, fake_images, [], [], self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, fake_images, [], [], self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_images = fake_images
            batch_imglens[0] = 1

        if (
            self.template.mm_plugin.audio_token is not None and sum(batch_audlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": AUDIO_PLACEHOLDER}]
            fake_audios = [np.zeros(1600)]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, [], [], fake_audios, self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, [], [], fake_audios, self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_audios = fake_audios
            batch_audlens[0] = 1

        if len(fake_input_ids) != 0:
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]

            batch_input_ids[0] = features[0]["input_ids"]

        if self.template.mm_plugin.timeseries_token is not None:
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                batch_images,
                batch_videos,
                batch_audios,
                batch_imglens,
                batch_vidlens,
                batch_audlens,
                batch_input_ids,
                self.processor,
                timeseries=batch_timeseries,
                ts_patch_sizes=batch_ts_patch_sizes,
            )
        else:
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                batch_images,
                batch_videos,
                batch_audios,
                batch_imglens,
                batch_vidlens,
                batch_audlens,
                batch_input_ids,
                self.processor
            )

        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: dict[str, torch.Tensor] = super().__call__(features)

        if self.get_rope_func is not None:
            rope_index_kwargs = {
                "input_ids": features["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": (features["attention_mask"] >= 1).float(),
            }
            if "second_per_grid_ts" in mm_inputs:  # for qwen2vl
                rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
            elif "video_second_per_grid" in mm_inputs:  # for qwen2.5 omni
                rope_index_kwargs["second_per_grids"] = mm_inputs.get("video_second_per_grid")

            if getattr(self.model.config, "model_type", None) == "qwen2_5_omni_thinker":  # for qwen2.5 omni
                rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
                feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
                if feature_attention_mask is not None:  # FIXME: need to get video image lengths
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    rope_index_kwargs["audio_seqlens"] = audio_feature_lengths  # prepare for input

                features["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
                features["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(
                    dim=-1
                ).unsqueeze(-1)
            else:  # for qwen2vl
                features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)

        if (
            self.model is not None
            and getattr(self.model.config, "model_type", None)
            in ["glm4v", "qwen2_vl", "qwen2_5_vl", "qwen2_5_omni_thinker"]
            and ("position_ids" not in features or features["position_ids"].dim() != 3)
        ):
            raise ValueError("Qwen2-VL/Qwen2.5-Omni model requires 3D position ids for mrope.")

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for 4d attention mask."""

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        for key, value in features.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)

        return features


def icot_lambda_distribution(removal_smoothing_lambda: float, truncate_length: int = 100) -> torch.Tensor:
    r"""Removal-offset distribution P(o) ∝ exp(-λ·o), truncated at ``truncate_length``; λ = inf gives o = 0."""
    if removal_smoothing_lambda == float("inf"):
        dist = torch.zeros(truncate_length)
        dist[0] = 1
        return dist
    positions = torch.arange(truncate_length)
    dist = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
    cum_prob = dist.sum()
    assert cum_prob <= 1
    dist[-1] = dist[-1] + (1 - cum_prob)
    return dist


@dataclass
class IcotDataCollator(SFTDataCollatorWith4DAttentionMask):
    r"""SFT collator that removes the first ``scheduled_to_remove`` (+ a random offset) CoT tokens of each row."""

    scheduled_to_remove: int = 0
    remove_all: bool = False
    removal_smoothing_lambda: float = float("inf")

    def __post_init__(self):
        super().__post_init__()
        self._lambda_distribution = icot_lambda_distribution(self.removal_smoothing_lambda)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        offsets = torch.multinomial(self._lambda_distribution, len(features), replacement=True).tolist()
        for feature, offset in zip(features, offsets):
            start = feature.pop("icot_cot_start")
            cot_len = feature.pop("icot_cot_len")
            n = cot_len if self.remove_all else min(self.scheduled_to_remove + offset, cot_len)
            if n > 0:
                for key in ("input_ids", "attention_mask", "labels"):
                    feature[key] = feature[key][:start] + feature[key][start + n :]

        return super().__call__(features)


@dataclass
class LaprasDataCollatorWithPadding:
    r"""Pads the student prompt (left), the student answer and the teacher sequence (right)."""

    tokenizer: Any
    label_pad_token_id: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None
    template: Any = None
    processor: Any = None

    def _compute_max_len(self, sequences: list[list[int]]) -> int:
        max_len = max(len(s) for s in sequences)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        return max_len

    def _pad_right(self, sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = self._compute_max_len(sequences)
        padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def _pad_left(self, sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        """Left-pad so last token is always rightmost (for encoder with bot_id)."""
        max_len = self._compute_max_len(sequences)
        padded = [[pad_value] * (max_len - len(s)) + s for s in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id

        batch = {}

        batch["encoder_input_ids"] = self._pad_left(
            [f["encoder_input_ids"] for f in features], pad_id
        )
        batch["encoder_attention_mask"] = batch["encoder_input_ids"].ne(pad_id).long()

        batch["decoder_input_ids"] = self._pad_right(
            [f["decoder_input_ids"] for f in features], pad_id
        )
        batch["labels"] = self._pad_right(
            [f["labels"] for f in features], self.label_pad_token_id
        )

        batch["ref_input_ids"] = self._pad_right(
            [f["ref_input_ids"] for f in features], pad_id
        )
        batch["ref_labels"] = self._pad_right(
            [f["ref_labels"] for f in features], self.label_pad_token_id
        )
        batch["ref_attention_mask"] = batch["ref_input_ids"].ne(pad_id).long()

        batch["ref_answer_position"] = torch.tensor(
            [f["ref_answer_position"] for f in features], dtype=torch.long
        )
        batch["model_answer_position"] = torch.tensor(
            [f["model_answer_position"] for f in features], dtype=torch.long
        )

        if self.template is not None and self.template.mm_plugin.timeseries_token is not None:
            batch_timeseries = []
            batch_ts_patch_sizes = []
            is_per_sample_ts = isinstance(self.template.mm_plugin, (SlipPlugin, FlamingoPlugin))
            for f in features:
                ts = f.pop("timeseries", None) or []
                ts_ps = f.pop("ts_patch_sizes", None) or []
                if is_per_sample_ts:
                    if ts:
                        batch_timeseries.append(ts)
                        batch_ts_patch_sizes.append(ts_ps[0] if ts_ps else None)
                else:
                    batch_timeseries.extend(ts)
                    batch_ts_patch_sizes.extend(ts_ps)

            if batch_timeseries:
                mm_inputs = self.template.mm_plugin.get_mm_inputs(
                    [], [], [],  # images, videos, audios
                    [0] * len(features), [0] * len(features), [0] * len(features),
                    [f.get("encoder_input_ids", []) for f in features],
                    self.processor,
                    timeseries=batch_timeseries,
                    ts_patch_sizes=batch_ts_patch_sizes,
                )
                batch.update(mm_inputs)

        return batch


@dataclass
class CoconutDataCollatorWithPadding(LaprasDataCollatorWithPadding):
    r"""COCONUT collator: the decoder keeps the CoT steps after the first ``n_skip_steps``, then the answer."""

    stage: int = 0
    n_skip_steps: int = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, Any] = {}

        batch["encoder_input_ids"] = self._pad_left(
            [f["encoder_input_ids"] for f in features], pad_id
        )
        batch["encoder_attention_mask"] = batch["encoder_input_ids"].ne(pad_id).long()

        decoder_ids: list[list[int]] = []
        for f in features:
            kept = f["cot_step_ids"][self.n_skip_steps :]
            flat_cot = [t for step in kept for t in step]
            decoder_ids.append(list(f["decoder_prefix_ids"]) + flat_cot + list(f["answer_ids"]))

        batch["decoder_input_ids"] = self._pad_right(decoder_ids, pad_id)
        # Mask padding by position, not by id: pad == eos on these backbones.
        batch["labels"] = self._pad_right(decoder_ids, self.label_pad_token_id)

        if self.template is not None and self.template.mm_plugin.timeseries_token is not None:
            batch_timeseries = []
            batch_ts_patch_sizes = []
            is_per_sample_ts = isinstance(
                self.template.mm_plugin, (SlipPlugin, FlamingoPlugin)
            )
            for f in features:
                ts = f.pop("timeseries", None) or []
                ts_ps = f.pop("ts_patch_sizes", None) or []
                if is_per_sample_ts:
                    if ts:
                        batch_timeseries.append(ts)
                        batch_ts_patch_sizes.append(ts_ps[0] if ts_ps else None)
                else:
                    batch_timeseries.extend(ts)
                    batch_ts_patch_sizes.extend(ts_ps)

            if batch_timeseries:
                mm_inputs = self.template.mm_plugin.get_mm_inputs(
                    [], [], [],
                    [0] * len(features), [0] * len(features), [0] * len(features),
                    [f.get("encoder_input_ids", []) for f in features],
                    self.processor,
                    timeseries=batch_timeseries,
                    ts_patch_sizes=batch_ts_patch_sizes,
                )
                batch.update(mm_inputs)

        return batch
