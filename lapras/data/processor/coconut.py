"""COCONUT data processor: tokenizes the CoT step by step for the curriculum."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .lapras import LaprasDatasetProcessor, _STEP_SEPARATOR_RE


logger = logging.get_logger(__name__)

# One <|bot|>...<|eot|> step, together with the tool span that follows it.
_BOT_STEP_RE = re.compile(
    r"<\|bot\|>(?:(?!<\|bot\|>).)*?<\|eot\|>"
    r"(?:\s*<timeseries_selection_tool>.*?</timeseries_selection_tool>)?",
    re.DOTALL,
)


def split_cot_steps(cot_text: str) -> list[str]:
    """Split a CoT into reasoning steps: <|bot|>...<|eot|> blocks, ``---`` blocks, or sentences."""
    if not cot_text:
        return []

    matches = list(_BOT_STEP_RE.finditer(cot_text))
    if matches:
        residue = _BOT_STEP_RE.sub("", cot_text).strip()
        if residue:
            raise ValueError(
                "[COCONUT] split_cot_steps: <|bot|> blocks do not tile the CoT — "
                f"{len(residue)} chars of text sit outside any step and would be "
                f"silently dropped. Residue: {residue[:300]!r}"
            )
        return [m.group(0).strip() for m in matches]

    parts = _STEP_SEPARATOR_RE.split(cot_text)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]

    sentences = [s.strip() for s in cot_text.split(". ") if s.strip()]
    return [s if s.endswith(".") else s + "." for s in sentences]


@dataclass
class CoconutDatasetProcessor(LaprasDatasetProcessor):
    r"""Emit the prompt, the tokenized CoT steps and the answer; the collator applies the stage."""

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        eos_id = self.tokenizer.eos_token_id
        n_steps_seen: list[int] = []

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

            steps = split_cot_steps(cot_text)
            if not steps:
                continue

            cot_step_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in steps]

            answer_anchor_ids = self.answer_prompt_ids[0]
            answer_body = answer_text[len(self.data_args.answer_anchor):].strip()
            answer_body_ids = (
                self.tokenizer.encode(answer_body, add_special_tokens=False) if answer_body else []
            )
            answer_ids = answer_anchor_ids + answer_body_ids + [eos_id]

            # prompt + <|bot|> + thoughts + <|eot|> + steps[stage:] + answer + eos
            decoder_prefix_ids = [self.eot_id]
            encoder_input_ids = prompt_ids + [self.bot_id]

            model_inputs["encoder_input_ids"].append(encoder_input_ids)
            model_inputs["decoder_prefix_ids"].append(decoder_prefix_ids)
            model_inputs["cot_step_ids"].append(cot_step_ids)
            model_inputs["answer_ids"].append(answer_ids)
            n_steps_seen.append(len(cot_step_ids))

            model_inputs["timeseries"].append(
                examples["_timeseries"][i] if "_timeseries" in examples else None
            )
            ts_data = (examples["_timeseries"][i] or []) if "_timeseries" in examples else []
            ts_ps = examples["_ts_patch_size"][i] if "_ts_patch_size" in examples else None
            model_inputs["ts_patch_sizes"].append(
                [ts_ps] * len(ts_data) if ts_ps is not None else None
            )

        if n_steps_seen and not hasattr(self, "_step_stats_logged"):
            self._step_stats_logged = True
            logger.info_rank0(
                f"[COCONUT] step counts in first shard: min={min(n_steps_seen)} "
                f"max={max(n_steps_seen)} mean={sum(n_steps_seen) / len(n_steps_seen):.2f} "
                f"(n={len(n_steps_seen)}). max_latent_stage should be >= max to let the "
                f"curriculum internalize every step."
            )

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        dec = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731
        print("encoder_input_ids:\n{}".format(example["encoder_input_ids"]))
        print("encoder (decoded):\n{}".format(dec(example["encoder_input_ids"])))
        print("decoder_prefix_ids: {} -> {!r}".format(
            example["decoder_prefix_ids"], dec(example["decoder_prefix_ids"])
        ))
        print("num_cot_steps: {}".format(len(example["cot_step_ids"])))
        for k, step in enumerate(example["cot_step_ids"]):
            print("  step[{}] ({} tok): {!r}".format(k, len(step), dec(step)))
        print("answer_ids: {} -> {!r}".format(example["answer_ids"], dec(example["answer_ids"])))
        print(
            "\nAt curriculum stage k, decoder = decoder_prefix + concat(step[k:]) + answer.\n"
            "  stage 0            -> full CoT (== cot_sft)\n"
            "  stage {}  -> answer only (all steps internalized)".format(
                len(example["cot_step_ids"])
            )
        )
