"""Lapras data processor: builds the student and teacher sequences of each example."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .supervised import SupervisedDatasetProcessor


logger = logging.get_logger(__name__)

# Horizontal-rule separator between CoT steps ("---" on its own line).
_STEP_SEPARATOR_RE = re.compile(r"\n[ \t]*-{3,}[ \t]*\n", re.MULTILINE)


def drop_last_cot_step(cot_text: str) -> str:
    """Drop the last CoT step, which restates the answer."""
    parts = _STEP_SEPARATOR_RE.split(cot_text)
    if len(parts) > 1:
        return "\n\n---\n\n".join(parts[:-1]).rstrip()
    if "<|bot|>" in cot_text:
        return cot_text[: cot_text.rfind("<|bot|>")].rstrip()
    sentences = cot_text.split(". ")[:-1]
    return ". ".join(sentences) + "." if sentences else ""


def get_answer_token_position(tokens: list[int], answer_prompt_ids: list[list[int]]) -> Optional[int]:
    """Index of the first token after the last answer anchor (the distillation anchor), or None."""
    result = None
    for pattern in answer_prompt_ids:
        pattern_len = len(pattern)
        for i in range(len(tokens) - pattern_len, -1, -1):
            if tokens[i : i + pattern_len] == pattern:
                result = i + pattern_len
                break  # last occurrence of this pattern
    return result


@dataclass
class LaprasDatasetProcessor(SupervisedDatasetProcessor):
    def __post_init__(self):
        self.bot_id = self.tokenizer.convert_tokens_to_ids("<|bot|>")
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|eot|>")
        for token, token_id in (("<|bot|>", self.bot_id), ("<|eot|>", self.eot_id)):
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    f"{token} is not a registered token. Pass "
                    '`--add_special_tokens "<|bot|>,<|eot|>" --resize_vocab True`.'
                )

        self.answer_prompt_ids: list[list[int]] = [self.tokenizer.encode(self.data_args.answer_anchor, add_special_tokens=False)]

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        remove_eos = self.data_args.lapras_remove_eos
        eos_id = self.tokenizer.eos_token_id

        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            result = super()._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                timeseries=(examples["_timeseries"][i] or []) if "_timeseries" in examples else [],
            )
            if result[0] is None:
                continue
            input_ids, labels = result

            prompt_len = 0
            for j, lbl in enumerate(labels):
                if lbl != IGNORE_INDEX:
                    prompt_len = j
                    break

            prompt_ids = list(input_ids[:prompt_len])

            response_ids = input_ids[prompt_len:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            eos_token = self.tokenizer.eos_token
            if eos_token and response_text.endswith(eos_token):
                response_text = response_text[: -len(eos_token)].rstrip()

            anchor_idx = response_text.rfind(self.data_args.answer_anchor)
            if anchor_idx < 0:
                continue

            cot_text = response_text[:anchor_idx].strip()
            answer_text = response_text[anchor_idx:].strip()
            if cot_text:
                cot_text = drop_last_cot_step(cot_text)

            # Tokenized separately so the answer starts with the exact anchor ids.
            cot_ids = self.tokenizer.encode(cot_text, add_special_tokens=False) if cot_text else []
            answer_body = answer_text[len(self.data_args.answer_anchor) :].strip()
            answer_body_ids = self.tokenizer.encode(answer_body, add_special_tokens=False) if answer_body else []
            answer_ids = self.answer_prompt_ids[0] + answer_body_ids + [eos_id]

            decoder_prefix = [self.eot_id] if remove_eos else [self.eot_id, eos_id]
            if not remove_eos:
                prompt_ids = prompt_ids + [eos_id]
                cot_ids = cot_ids + [eos_id] if cot_ids else []

            decoder_input_ids = decoder_prefix + answer_ids
            ref_input_ids = prompt_ids + cot_ids + answer_ids
            ref_labels = [IGNORE_INDEX] * len(prompt_ids) + cot_ids + answer_ids
            encoder_input_ids = prompt_ids + [self.bot_id]

            model_inputs["encoder_input_ids"].append(encoder_input_ids)
            model_inputs["decoder_input_ids"].append(decoder_input_ids)
            model_inputs["labels"].append(list(decoder_input_ids))
            model_inputs["ref_input_ids"].append(ref_input_ids)
            model_inputs["ref_labels"].append(ref_labels)
            model_inputs["ref_answer_position"].append(get_answer_token_position(ref_input_ids, self.answer_prompt_ids))
            model_inputs["model_answer_position"].append(
                get_answer_token_position(decoder_input_ids, self.answer_prompt_ids)
            )

            # Raw time series; the collator encodes them.
            model_inputs["timeseries"].append(examples["_timeseries"][i] if "_timeseries" in examples else None)
            ts_data = (examples["_timeseries"][i] or []) if "_timeseries" in examples else []
            ts_ps = examples["_ts_patch_size"][i] if "_ts_patch_size" in examples else None
            model_inputs["ts_patch_sizes"].append([ts_ps] * len(ts_data) if ts_ps is not None else None)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731
        print("encoder (student prompt):\n{}".format(decode(example["encoder_input_ids"])))
        print("decoder (student answer):\n{}".format(decode(example["decoder_input_ids"])))
        print("ref_input_ids length: {}".format(len(example["ref_input_ids"])))
        valid_ref_labels = [t for t in example["ref_labels"] if t != IGNORE_INDEX]
        print("teacher target (CoT + answer):\n{}".format(decode(valid_ref_labels)))
        for name, pos_key, ids_key in (
            ("teacher", "ref_answer_position", "ref_input_ids"),
            ("student", "model_answer_position", "decoder_input_ids"),
        ):
            pos = example[pos_key]
            if pos is None:
                print(f"{name} anchor: None (anchor not found!)")
                continue
            ids = example[ids_key]
            print(f"{name} anchor={pos} token={decode(ids[pos : pos + 1])!r}")
