"""iCoT-SI data processor (Deng et al., 2024): the SFT processor plus the token span of the CoT."""

from dataclasses import dataclass
from typing import Any

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .supervised import SupervisedDatasetProcessor


logger = logging.get_logger(__name__)


@dataclass
class IcotDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not getattr(self.data_args, "use_cot", True):
            raise ValueError(
                "[iCoT] stage=icot needs the CoT in the response (use_cot=True); with "
                "use_cot=False there is nothing to internalize."
            )

        model_inputs = super().preprocess_dataset(examples)
        cot_lens: list[int] = []

        for input_ids, labels in zip(model_inputs["input_ids"], model_inputs["labels"]):
            cot_start = next((j for j, lbl in enumerate(labels) if lbl != IGNORE_INDEX), None)
            if cot_start is None:
                raise ValueError("[iCoT] row has no supervised tokens (response truncated by cutoff_len?)")

            # The CoT ends at the last token from which the decoded response starts with the anchor.
            response_ids = input_ids[cot_start:]
            cot_len = None
            for j in range(len(response_ids) - 1, -1, -1):
                if self.tokenizer.decode(response_ids[j:], skip_special_tokens=False).lstrip().startswith(self.data_args.answer_anchor):
                    cot_len = j
                    break
            if cot_len is None:
                raise ValueError(
                    f"[iCoT] response has no token starting {self.data_args.answer_anchor!r}, so the CoT span is undefined. "
                    f"Response tail: {self.tokenizer.decode(response_ids[-60:], skip_special_tokens=False)!r}"
                )

            model_inputs["icot_cot_start"].append(cot_start)
            model_inputs["icot_cot_len"].append(cot_len)
            cot_lens.append(cot_len)

        if cot_lens and not hasattr(self, "_cot_stats_logged"):
            self._cot_stats_logged = True
            s = sorted(cot_lens)
            logger.info_rank0(
                f"[iCoT] CoT tokens in first shard: n={len(s)} p50={s[len(s) // 2]} "
                f"p90={s[int(0.9 * (len(s) - 1))]} p99={s[int(0.99 * (len(s) - 1))]} max={s[-1]}. "
                f"icot_remove_all_when_remove_beyond is normally set near p99."
            )

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        super().print_data_example(example)
        start, n = example["icot_cot_start"], example["icot_cot_len"]
        dec = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731
        print(f"icot_cot_start={start} icot_cot_len={n}")
        print(f"CoT span (removed from the left over training):\n{dec(example['input_ids'][start:start + n])!r}")
        print(f"kept at full removal:\n{dec(example['input_ids'][start + n:])!r}")
