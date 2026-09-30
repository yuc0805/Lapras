"""COCONUT training workflow — mirrors run_lapras, minus the teacher."""

from typing import TYPE_CHECKING, Optional

from ...data import CoconutDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import build_run_config, create_modelcard_and_push
from .curriculum import CoconutStageCallback
from .trainer import CoconutTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_coconut(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """Run the COCONUT (Chain of Continuous Thought) training stage."""
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template, model_args, data_args, training_args, stage="coconut", **tokenizer_module
    )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    c_thought = finetuning_args.coconut_c_thought
    epochs_per_stage = finetuning_args.coconut_epochs_per_stage
    max_latent_stage = finetuning_args.coconut_max_latent_stage

    epochs_needed = (max_latent_stage + 1) * epochs_per_stage
    if training_args.num_train_epochs < epochs_needed:
        raise ValueError(
            f"[COCONUT] num_train_epochs={training_args.num_train_epochs} cannot reach the "
            f"terminal stage: max_latent_stage={max_latent_stage} with "
            f"epochs_per_stage={epochs_per_stage} needs >= {epochs_needed} epochs "
            f"(stages 0..{max_latent_stage}). Upstream trains well past that (GSM8k: 25 epochs "
            f"for a 4-stage schedule) so the terminal stage gets most of the training."
        )

    if training_args.local_process_index == 0:
        logger.info_rank0(
            f"[COCONUT] curriculum: c_thought={c_thought} epochs_per_stage={epochs_per_stage} "
            f"max_latent_stage={max_latent_stage} -> stages 0..{max_latent_stage} over the first "
            f"{epochs_needed} epochs, terminal K={c_thought * max_latent_stage}, then "
            f"{training_args.num_train_epochs - epochs_needed:.0f} epoch(s) at the terminal stage."
        )

    data_collator = CoconutDataCollatorWithPadding(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.do_train else None,
        template=template,
        processor=tokenizer_module.get("processor"),
        stage=0,
    )

    trainer = CoconutTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        run_config=build_run_config(model_args, data_args, finetuning_args),
        **tokenizer_module,
        **dataset_module,
    )

    trainer.add_callback(
        CoconutStageCallback(
            collator=data_collator,
            trainer=trainer,
            c_thought=c_thought,
            epochs_per_stage=epochs_per_stage,
            max_latent_stage=max_latent_stage,
            reset_optimizer=finetuning_args.coconut_reset_optimizer,
        )
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
