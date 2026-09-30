"""iCoT-SI token-removal schedule: s(t) = floor(cutoff * t / (ramp_epochs * steps_per_epoch))."""

import json
import os
from typing import TYPE_CHECKING

from transformers import TrainerCallback

from ...extras import logging


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

    from ...data import IcotDataCollator


logger = logging.get_logger(__name__)


def scheduled_to_remove(step: int, cutoff: int, ramp_epochs: int, steps_per_epoch: int) -> int:
    """Leading CoT tokens scheduled for removal before optimizer step ``step`` (0-based)."""
    return cutoff * int(step) // (ramp_epochs * steps_per_epoch)


class IcotRemovalCallback(TrainerCallback):
    r"""Advance the collator's removal count every optimizer step."""

    def __init__(
        self,
        collator: "IcotDataCollator",
        cutoff: int,
        ramp_epochs: int,
        removal_smoothing_lambda: float,
        reset_every_epochs: int = 0,
    ):
        self.collator = collator
        self.cutoff = cutoff
        self.ramp_epochs = ramp_epochs
        self.removal_smoothing_lambda = removal_smoothing_lambda
        self.reset_every_epochs = reset_every_epochs
        self.steps_per_epoch: int = 0
        self._last_reset_epoch = 0

    def _apply(self, step: int) -> None:
        s = scheduled_to_remove(step, self.cutoff, self.ramp_epochs, self.steps_per_epoch)
        remove_all = s >= self.cutoff
        if remove_all and not self.collator.remove_all:
            logger.info_rank0(f"[iCoT] step {step}: scheduled removal reached {self.cutoff} -> removing ALL CoT tokens")
        self.collator.scheduled_to_remove = min(s, self.cutoff)
        self.collator.remove_all = remove_all

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        epochs = args.num_train_epochs
        if epochs != int(epochs) or state.max_steps % int(epochs) != 0:
            raise ValueError(
                f"[iCoT] need integer epochs and max_steps divisible by them; got "
                f"num_train_epochs={epochs}, max_steps={state.max_steps}"
            )
        self.steps_per_epoch = state.max_steps // int(epochs)
        self._last_reset_epoch = state.global_step // self.steps_per_epoch
        logger.info_rank0(
            f"[iCoT] schedule: {self.steps_per_epoch} steps/epoch, Δ={self.cutoff / self.ramp_epochs:.2f} tokens/epoch "
            f"({self.cutoff / (self.ramp_epochs * self.steps_per_epoch):.3f}/step), remove-all at step "
            f"{self.ramp_epochs * self.steps_per_epoch} (epoch {self.ramp_epochs}) of {state.max_steps}; "
            f"λ={self.removal_smoothing_lambda}; optimizer reset every {self.reset_every_epochs or 'never'} epoch(s)"
        )
        self._apply(state.global_step)

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._apply(state.global_step)

    def on_epoch_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        epoch = state.global_step // self.steps_per_epoch
        logger.info_rank0(
            f"[iCoT] epoch {epoch}: removing {'ALL' if self.collator.remove_all else self.collator.scheduled_to_remove} "
            f"leading CoT token(s)"
        )
        optimizer = kwargs.get("optimizer")
        if (
            self.reset_every_epochs
            and epoch > self._last_reset_epoch
            and epoch % self.reset_every_epochs == 0
            and optimizer is not None
        ):
            optimizer.state.clear()
            self._last_reset_epoch = epoch
            logger.info_rank0(f"[iCoT] optimizer state reset at epoch {epoch}")

    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return
        meta = {
            "stage": "icot",
            "icot_remove_all_when_remove_beyond": self.cutoff,
            "icot_ramp_epochs": self.ramp_epochs,
            "icot_remove_per_epoch": self.cutoff / self.ramp_epochs,
            "icot_removal_smoothing_lambda": self.removal_smoothing_lambda,
            "icot_reset_optimizer_every_epochs": self.reset_every_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "global_step": state.global_step,
            "all_cot_removed_at_save": self.collator.remove_all,
        }
        with open(os.path.join(ckpt_dir, "icot_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
