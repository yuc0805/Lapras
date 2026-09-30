"""iCoT-SI training workflow: CoT-SFT with a collator that removes leading CoT tokens."""

import math
from typing import TYPE_CHECKING, Optional

from ...data import IcotDataCollator, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..sft.trainer import CustomSeq2SeqTrainer
from ..trainer_utils import build_run_config, create_modelcard_and_push, root_save_would_duplicate
from .schedule import IcotRemovalCallback


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_icot(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """Run the iCoT-SI (Stepwise Internalization) training stage."""
    cutoff = finetuning_args.icot_remove_all_when_remove_beyond
    ramp_epochs = finetuning_args.icot_ramp_epochs
    lam = finetuning_args.icot_removal_smoothing_lambda
    reset_every = finetuning_args.icot_reset_optimizer_every_epochs

    if cutoff <= 0 or ramp_epochs <= 0:
        raise ValueError(
            f"[iCoT] --icot_remove_all_when_remove_beyond ({cutoff}) and --icot_ramp_epochs ({ramp_epochs}) "
            "must both be > 0 — there is no sensible default for a dataset's CoT length."
        )
    if ramp_epochs > training_args.num_train_epochs:
        raise ValueError(
            f"[iCoT] icot_ramp_epochs={ramp_epochs} > num_train_epochs={training_args.num_train_epochs}: "
            "the CoT would never be fully removed, so the checkpoint would not be the answer-only model "
            "the eval assumes."
        )
    if reset_every < 0 or not (lam > 0 or math.isinf(lam)):
        raise ValueError(f"[iCoT] bad reset_every={reset_every} / removal_smoothing_lambda={lam}")
    if training_args.dataloader_num_workers != 0:
        raise ValueError(
            f"[iCoT] dataloader_num_workers={training_args.dataloader_num_workers}: the collator's removal "
            "count is updated in the main process between steps; worker processes hold a stale copy and "
            "would train at removal 0 forever. Use 0."
        )
    if training_args.do_eval:
        raise ValueError("[iCoT] in-trainer eval is not wired; score checkpoints with evaluation/evaluate.py.")
    if data_args.packing:
        raise ValueError("[iCoT] packing would merge rows before the collator can cut each row's CoT.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="icot", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = IcotDataCollator(
        template=template,
        model=model,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        removal_smoothing_lambda=lam,
        **tokenizer_module,
    )

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        run_config=build_run_config(model_args, data_args, finetuning_args),
        **dataset_module,
        **tokenizer_module,
    )
    trainer.add_callback(
        IcotRemovalCallback(
            collator=data_collator,
            cutoff=cutoff,
            ramp_epochs=ramp_epochs,
            removal_smoothing_lambda=lam,
            reset_every_epochs=reset_every,
        )
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        if not root_save_would_duplicate(trainer):
            trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
