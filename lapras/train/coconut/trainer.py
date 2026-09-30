"""COCONUT trainer (Hao et al., 2024; github.com/facebookresearch/coconut)."""

import os
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.nn as nn
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..lapras.trainer import get_input_embedding_layer
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, save_run_config


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CoconutTrainer(Trainer):
    r"""Latent loop trained with CE only; the number of thoughts K follows the curriculum stage."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        run_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.run_config = run_config
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # K of the current curriculum stage, set by CoconutStageCallback.
        self.num_latent = 0
        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ):
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        if self.is_world_process_zero():
            save_run_config(output_dir if output_dir is not None else self.args.output_dir, self.run_config)

    @override
    def _save_checkpoint(self, model, trial):
        result = super()._save_checkpoint(model, trial)
        if self.is_world_process_zero():
            save_run_config(
                os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}"), self.run_config
            )
        return result

    @override
    def compute_loss(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, "torch.Tensor"],
        *args,
        **kwargs,
    ) -> Union["torch.Tensor", tuple["torch.Tensor", dict[str, Any]]]:
        """One COCONUT step: encode -> K continuous thoughts -> decode text -> CE."""
        encoder_input_ids = inputs["encoder_input_ids"]  # prompt + <|bot|>
        decoder_input_ids = inputs["decoder_input_ids"]  # <|eot|> + remaining CoT steps + answer
        labels = inputs["labels"]
        encoder_attention_mask = inputs.get("encoder_attention_mask")
        timeseries = inputs.get("timeseries")
        ts_patch_sizes = inputs.get("ts_patch_sizes")
        sensor_attn_mask = inputs.get("sensor_attn_mask")
        time_index = inputs.get("time_index")

        _inner = getattr(model, "module", model)
        _model_type = getattr(getattr(_inner, "config", None), "model_type", None)

        ts_kwargs: dict[str, Any] = {"timeseries": timeseries, "ts_patch_sizes": ts_patch_sizes}
        if (
            _model_type in ("slip", "opentslm_flamingo")
            or sensor_attn_mask is not None
            or time_index is not None
        ):
            ts_kwargs = {
                "timeseries": timeseries,
                "sensor_attn_mask": sensor_attn_mask,
                "time_index": time_index,
            }

        embed_layer = get_input_embedding_layer(model)

        encoder_outputs = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            use_cache=True,
            output_hidden_states=False,
            **ts_kwargs,
        )
        past_key_values = encoder_outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError(
                "[COCONUT] encoder returned no KV cache — use_cache was forced off "
                "(gradient checkpointing?). The latent loop reuses this cache and cannot "
                "run without it. Add --disable_gradient_checkpointing True."
            )

        # ChatTS returns the attention mask of the TS-expanded sequence.
        if getattr(encoder_outputs, "attention_mask", None) is not None:
            encoder_attention_mask = encoder_outputs.attention_mask

        # Continuous thoughts: the last hidden state is fed back unchanged.
        latent_embd = encoder_outputs.last_hidden_state[:, -1, :].unsqueeze(1)
        num_latent = self.num_latent
        for _ in range(num_latent):
            outputs = model(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Decode the remaining CoT steps and the answer.
        outputs = model(
            inputs_embeds=embed_layer(decoder_input_ids),
            use_cache=False,
            output_hidden_states=False,
            past_key_values=past_key_values,
        )

        logits = outputs.logits
        ce_loss = self.ce_loss_fct(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )

        if not hasattr(self, "_coconut_shape_logged"):
            self._coconut_shape_logged = True
            logger.info_rank0(
                f"[COCONUT] step shape: encoder={tuple(encoder_input_ids.shape)} "
                f"K={num_latent} decoder={tuple(decoder_input_ids.shape)} "
                f"supervised_tokens={int((labels[:, 1:] != IGNORE_INDEX).sum())} "
                f"(collator stage={getattr(self.data_collator, 'stage', '?')})"
            )

        return ce_loss
