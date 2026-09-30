# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import os
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.packages import is_transformers_version_greater_than
from ...model.model_utils.timeseries import get_timeseries_learning_rate, maybe_apply_timeseries_sft_lr
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, save_run_config


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Seq2SeqTrainer with the custom optimizer/scheduler hooks, used by the SFT and iCoT stages."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        run_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False
            self.add_callback(SaveProcessorCallback(processor))

        self.finetuning_args = finetuning_args
        self.run_config = run_config

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        optimizer = super().create_optimizer()
        if optimizer is not None:
            maybe_apply_timeseries_sft_lr(optimizer, self.model, self.finetuning_args)
        return optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def log(self, logs: dict[str, Any], *args, **kwargs) -> None:
        if logs is not None and self.finetuning_args.timeseries_sft_lr is not None:
            ts_lr = get_timeseries_learning_rate(getattr(self, "optimizer", None))
            if ts_lr is not None and "ts_encoder_learning_rate" not in logs:
                logs["ts_encoder_learning_rate"] = ts_lr

        super().log(logs, *args, **kwargs)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        if self.is_world_process_zero():
            save_run_config(output_dir if output_dir is not None else self.args.output_dir, self.run_config)

    @override
    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.is_world_process_zero():
            save_run_config(
                os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}"), self.run_config
            )
