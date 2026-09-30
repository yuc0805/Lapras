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

from typing import TYPE_CHECKING, Any, Optional

import torch.distributed as dist

from ..extras import logging
from ..hparams import get_train_args, read_args
from .callbacks import LogCallback, ReporterCallback
from .coconut import run_coconut
from .icot import run_icot
from .lapras import run_lapras
from .sft import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks = callbacks or []
    callbacks.append(LogCallback())
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    if finetuning_args.stage == "sft":  # No-CoT (use_cot=False) and CoT (use_cot=True) SFT
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "lapras":
        run_lapras(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "coconut":
        run_coconut(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "icot":
        run_icot(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError(f"Unknown stage: {finetuning_args.stage}. Expected one of: sft, lapras, coconut, icot.")

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")
