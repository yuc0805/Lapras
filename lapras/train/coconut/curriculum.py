"""COCONUT curriculum: at stage k the first k CoT steps are replaced by continuous thoughts."""

from typing import TYPE_CHECKING, Optional

from transformers import TrainerCallback

from ...extras import logging


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


logger = logging.get_logger(__name__)


def scheduled_stage(epoch: int, epochs_per_stage: int) -> int:
    """Curriculum stage of an epoch."""
    if epochs_per_stage <= 0:
        raise ValueError(f"[COCONUT] epochs_per_stage must be >= 1, got {epochs_per_stage}")
    return int(epoch) // int(epochs_per_stage)


def num_latent_for_stage(stage: int, c_thought: int, max_latent_stage: int) -> int:
    """Number of continuous thoughts K at a stage, capped at ``c_thought * max_latent_stage``."""
    return int(c_thought) * min(int(stage), int(max_latent_stage))


_SKIP_ALL = 10**9


def n_skip_for_stage(stage: int, max_latent_stage: int) -> int:
    """Number of leading CoT steps deleted at a stage; all of them past ``max_latent_stage``."""
    return _SKIP_ALL if int(stage) > int(max_latent_stage) else int(stage)


class CoconutStageCallback(TrainerCallback):
    r"""Set the collator's stage and the trainer's K each epoch; reset the optimizer on a stage change."""

    def __init__(
        self,
        collator,
        trainer,
        c_thought: int,
        epochs_per_stage: int,
        max_latent_stage: int,
        reset_optimizer: bool = True,
    ):
        self.collator = collator
        self.trainer = trainer
        self.c_thought = c_thought
        self.epochs_per_stage = epochs_per_stage
        self.max_latent_stage = max_latent_stage
        self.reset_optimizer = reset_optimizer
        self._current_stage: Optional[int] = None

    def _apply(self, stage: int, optimizer=None) -> None:
        k = num_latent_for_stage(stage, self.c_thought, self.max_latent_stage)
        n_skip = n_skip_for_stage(stage, self.max_latent_stage)
        changed = self._current_stage != stage
        self.collator.stage = stage
        self.collator.n_skip_steps = n_skip
        self.trainer.num_latent = k

        if changed:
            deleting = "ALL" if n_skip >= _SKIP_ALL else str(n_skip)
            logger.info_rank0(
                f"[COCONUT] stage {self._current_stage} -> {stage}: deleting first {deleting} "
                f"CoT step(s), K={k} continuous thought(s) "
                f"(c_thought={self.c_thought}, max_latent_stage={self.max_latent_stage})"
                + (" [terminal: answer-only CE]" if n_skip >= _SKIP_ALL else "")
            )
            if self.reset_optimizer and optimizer is not None and self._current_stage is not None:
                optimizer.state.clear()
                logger.info_rank0("[COCONUT] optimizer state reset at stage switch")

        self._current_stage = stage

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        self._apply(scheduled_stage(0, self.epochs_per_stage), optimizer=None)

    def on_epoch_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        self._apply(
            scheduled_stage(int(state.epoch or 0), self.epochs_per_stage),
            optimizer=kwargs.get("optimizer"),
        )
