# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ._instruction_rewrite import strip_cot_instructions
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


def _apply_no_cot(
    prompt: list[dict[str, str]],
    response: list[dict[str, str]],
    anchor: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Strip the reasoning instructions from the prompt and the CoT from the response."""
    new_prompt = [
        {**m, "content": strip_cot_instructions(m["content"])}
        if m.get("role") == "user"
        else m
        for m in prompt
    ]
    new_response = []
    for m in response:
        if m.get("role") == "assistant":
            content = m["content"]
            idx = content.rfind(anchor)
            if idx < 0:
                raise ValueError(
                    "use_cot=False requires the {anchor!r} anchor to know where the CoT ends, "
                    "but this assistant response has none, so there is nothing to trim.\n"
                    "  Fix the DATA (every response must contain a literal {anchor!r}) or do not "
                    "run this dataset with use_cot=False.\n"
                    "  Response was:\n{resp}".format(anchor=anchor, resp=content[:800])
                )

            content = content[idx:].strip()
            new_response.append({**m, "content": content})
        else:
            new_response.append(m)
    return new_prompt, new_response


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        timeseries: list[Any],
    ) -> tuple[list[int], list[int]]:
        if not getattr(self.data_args, "use_cot", True):
            prompt, response = _apply_no_cot(prompt, response, self.data_args.answer_anchor)
        if timeseries is not None and len(timeseries) > 0:
            messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor, timeseries=timeseries)
        else:
            messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                logger.warning_rank0(
                    f"Dropped lengthy example with length {total_length} > {self.data_args.cutoff_len}."
                )
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            # Drop examples whose <ts> placeholders were truncated (Qwen2.5 / Qwen3 <ts> token ids).
            if 151665 in source_ids:
                if source_len < len(source_ids) or target_len < len(target_ids) or len(timeseries) != list(source_ids).count(151665) or list(target_ids).count(151665) != 0:
                    logger.warning_rank0(f"[drop mismatch] {source_len=}, {target_len=}, {len(timeseries)=}, {list(source_ids).count(151665)=}, {list(source_ids).count(151666)=}, {list(target_ids).count(151665)=}")
                    return None, None
            elif 151669 in source_ids:
                if source_len < len(source_ids) or target_len < len(target_ids) or len(timeseries) != list(source_ids).count(151669) or list(target_ids).count(151669) != 0:
                    logger.warning_rank0(f"[drop mismatch] {source_len=}, {target_len=}, {len(timeseries)=}, {list(source_ids).count(151669)=}, {list(source_ids).count(151670)=}, {list(target_ids).count(151669)=}")
                    return None, None
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                timeseries=examples["_timeseries"][i] or [],
            )

            if input_ids is None or labels is None:
                logger.warning_rank0("Dropped invalid example due to mismatch between timeseries and special tokens.")
                continue

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            model_inputs["timeseries"].append(examples["_timeseries"][i])
            ts_data = examples["_timeseries"][i] or []
            ts_ps = examples["_ts_patch_size"][i]
            model_inputs["ts_patch_sizes"].append([ts_ps] * len(ts_data) if ts_ps is not None else None)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
