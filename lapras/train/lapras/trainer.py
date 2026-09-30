"""Lapras trainer: teacher-student self-distillation of chain-of-thought into continuous thoughts."""

import inspect
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX, PROJECTION_WEIGHTS_NAME
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, detached_state_dict, save_run_config
from .masking import SensorMaskApplier, find_sensor_patch_embed


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


def get_input_embedding_layer(model: nn.Module) -> nn.Module:
    """Return the token-embedding module of a (possibly DeepSpeed/DDP/PEFT-wrapped) model."""
    unwrapped = getattr(model, "module", model)  # DeepSpeed / DDP
    if hasattr(unwrapped, "get_base_model"):  # PEFT
        unwrapped = unwrapped.get_base_model()
    embed = unwrapped.get_input_embeddings()
    if embed is None:
        raise NotImplementedError(f"Cannot find the input embedding layer of {type(unwrapped).__name__}.")
    return embed


def build_projection(hidden_size: int, projection_dim: int, dropout: float) -> nn.Sequential:
    """The projection pi that maps a continuous thought back into the input-embedding space."""
    projection = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(hidden_size, projection_dim),
        nn.GELU(),
        nn.Linear(projection_dim, hidden_size),
    )
    projection.add_module("ln", nn.LayerNorm(hidden_size))
    return projection


class LaprasTrainer(Trainer):
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        run_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False
        self.finetuning_args = finetuning_args
        self.run_config = run_config
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        hidden_size = self.model.config.hidden_size
        self.num_latent = finetuning_args.lapras_num_latent
        self.distill_weight = finetuning_args.lapras_distill_weight
        self.teacher_weight = finetuning_args.lapras_teacher_weight

        # ChatTS merges the series into the token sequence; 
        # SLIP/Flamingo cross-attend to a cached encoding.
        self.ref_merges_ts = getattr(self.model.config, "ts_token_start_index", None) is not None

        probe_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        fwd_params = inspect.signature(type(probe_model).forward).parameters
        self.encode_logits_kwargs = {"logits_to_keep": 1} if "logits_to_keep" in fwd_params else {}

        self.projection = None
        if finetuning_args.lapras_use_projection:
            self.projection = build_projection(
                hidden_size, finetuning_args.lapras_projection_dim, finetuning_args.lapras_projection_dropout
            )
            self.model.lapras_projection = self.projection

        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.distill_loss_fct = nn.SmoothL1Loss()

        # Sensor-patch masking of the student (training only).
        self.mask_ratio = finetuning_args.lapras_mask_ratio
        self.mask_applier = None
        self.mask_backend: Optional[str] = None
        self.sensor_patch_embed: Optional[nn.Module] = None
        if self.mask_ratio > 0.0:
            self.sensor_patch_embed, self.mask_backend, sensor_hidden = find_sensor_patch_embed(self.model)
            self.mask_applier = SensorMaskApplier(hidden_dim=sensor_hidden, mask_ratio=self.mask_ratio)
            self.model.lapras_mask_applier = self.mask_applier
            logger.info_rank0(
                f"[Lapras] sensor masking: backend={self.mask_backend}, ratio={self.mask_ratio}, "
                f"mask-token dim={sensor_hidden}; the teacher sees the clean series."
            )

    @contextmanager
    def _masked_sensors(self, sensor_attn_mask: Optional[Any]):
        r"""Replace a fraction of the sensor patches with [MASK] in the forward passes run inside this block."""
        handle = None
        if self.mask_applier is not None and self.model.training:
            mask = None
            if self.mask_backend == "slip" and sensor_attn_mask is not None:
                mask = self.mask_applier.sample_mask(sensor_attn_mask)
            elif self.mask_backend == "flamingo" and sensor_attn_mask is not None:
                mask = self.mask_applier.sample_mask_flamingo(sensor_attn_mask)

            def hook(_module, _args, output):
                if self.mask_backend == "chatts":
                    return self.mask_applier.apply_chatts_mask(output)
                if mask is None:
                    raise ValueError(f"The {self.mask_backend} batch has a time series but no `sensor_attn_mask`.")
                return self.mask_applier.apply_post_patch_embed(output, mask)

            handle = self.sensor_patch_embed.register_forward_hook(hook)
        try:
            yield
        finally:
            if handle is not None:
                handle.remove()

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""Add the sub-losses (averaged over gradient accumulation and reduced across ranks) to the logs."""
        train_eval = "train" if "loss" in logs else "eval"
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)

    def _remap_ref_to_expanded(
        self,
        ref_outputs: Any,
        ref_input_ids: "torch.Tensor",
        ref_answer_position: "torch.Tensor",
        ref_labels: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Map the teacher's anchor position and labels onto ChatTS's TS-expanded sequence."""
        exp_len = ref_outputs.logits.size(1)
        ntp = getattr(ref_outputs, "new_token_positions", None)
        if ntp is None:
            if exp_len != ref_input_ids.size(1):
                raise RuntimeError(
                    f"Teacher sequence expanded {ref_input_ids.size(1)} -> {exp_len} but the model returned no "
                    "`new_token_positions`, so the distillation anchor cannot be remapped."
                )
            return ref_answer_position, ref_labels

        inner = getattr(self.model, "module", self.model)
        ts_start = int(getattr(inner.config, "ts_token_start_index"))
        ts_end = int(getattr(inner.config, "ts_token_end_index"))

        new_answer_position = ntp.gather(1, ref_answer_position.unsqueeze(-1)).squeeze(-1)
        if bool((new_answer_position < 0).any()):
            raise RuntimeError("The teacher's answer anchor was remapped onto a padding slot.")

        keep = (ref_input_ids != ts_start) & (ref_input_ids != ts_end) & (ntp >= 0)
        b_idx, j_idx = keep.nonzero(as_tuple=True)
        new_labels = torch.full(
            (ref_labels.size(0), exp_len), IGNORE_INDEX, dtype=ref_labels.dtype, device=ref_labels.device
        )
        new_labels[b_idx, ntp[b_idx, j_idx]] = ref_labels[b_idx, j_idx]
        return new_answer_position, new_labels

    @override
    def compute_loss(self, model: "torch.nn.Module", inputs: dict[str, "torch.Tensor"], *args, **kwargs):
        encoder_input_ids = inputs["encoder_input_ids"]  # prompt + <|bot|>, left-padded
        decoder_input_ids = inputs["decoder_input_ids"]  # <|eot|> + answer, right-padded
        ref_input_ids = inputs["ref_input_ids"]  # prompt + CoT + answer, right-padded
        labels = inputs["labels"]
        ref_labels = inputs["ref_labels"]
        encoder_attention_mask = inputs.get("encoder_attention_mask")
        ref_answer_position = inputs["ref_answer_position"]
        model_answer_position = inputs["model_answer_position"]
        timeseries = inputs.get("timeseries")
        ts_patch_sizes = inputs.get("ts_patch_sizes")
        sensor_attn_mask = inputs.get("sensor_attn_mask")
        time_index = inputs.get("time_index")

        if self.ref_merges_ts:
            ts_kwargs: dict[str, Any] = {"timeseries": timeseries, "ts_patch_sizes": ts_patch_sizes}
        else:
            ts_kwargs = {"timeseries": timeseries, "sensor_attn_mask": sensor_attn_mask, "time_index": time_index}

        if not getattr(self, "_first_step_logged", False):
            self._first_step_logged = True
            if timeseries is None:
                logger.warning_rank0("[Lapras] the batch carries no time series; the model will not see sensor data.")

        embed_layer = get_input_embedding_layer(model)

        # Teacher: prompt + CoT + answer on the clean series.
        ref_outputs, ref_hidden_states = None, None
        if self.distill_weight > 0 or self.teacher_weight > 0:
            teacher_merges_ts = self.ref_merges_ts and timeseries is not None
            # ChatTS's merge reads attention_mask[:, 0] to pick the padding side.
            ref_attention_mask = inputs.get("ref_attention_mask") if teacher_merges_ts else None

            ref_outputs = model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                output_hidden_states=True,
                **ts_kwargs,
            )
            if teacher_merges_ts:
                ref_answer_position, ref_labels = self._remap_ref_to_expanded(
                    ref_outputs, ref_input_ids, ref_answer_position, ref_labels
                )

            ref_hidden_states = tuple(h.detach() for h in ref_outputs.hidden_states)

        # Student: encode prompt + <|bot|> on the masked series; the later forwards replay this encoding.
        with self._masked_sensors(sensor_attn_mask):
            encoder_outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                use_cache=True,
                output_hidden_states=False,
                **self.encode_logits_kwargs,
                **ts_kwargs,
            )
        past_key_values = encoder_outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError(
                "The encoder returned no KV cache (gradient checkpointing?). The latent loop reuses it; "
                "pass --disable_gradient_checkpointing True."
            )

        latent_embd = encoder_outputs.last_hidden_state[:, -1, :].unsqueeze(1)
        if self.projection is not None:
            latent_embd = self.projection(latent_embd)

        # Student: K continuous thoughts, then <|eot|> + answer.
        ce_loss = torch.tensor(0.0, device=encoder_input_ids.device)
        distill_loss = torch.tensor(0.0, device=encoder_input_ids.device)
        for i in range(self.num_latent):
            outputs = model(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if self.projection is not None:
                latent_embd = self.projection(latent_embd)

            if i == self.num_latent - 1:
                answer_embds = embed_layer(decoder_input_ids)
                outputs = model(
                    inputs_embeds=answer_embds,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )

                # Distillation at the first answer token, over all layers.
                if ref_hidden_states is not None:
                    for student_hidden, teacher_hidden in zip(outputs.hidden_states, ref_hidden_states):
                        ref_selected = teacher_hidden.gather(
                            1,
                            ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, teacher_hidden.size(-1)),
                        )
                        out_selected = student_hidden.gather(
                            1,
                            model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, student_hidden.size(-1)),
                        )
                        layer_loss = self.distill_loss_fct(out_selected, ref_selected)
                        layer_loss = layer_loss / ref_selected.std().detach()
                        distill_loss = distill_loss + layer_loss

                    distill_loss = distill_loss / len(outputs.hidden_states)

                # Student answer CE.
                logits = outputs.logits
                ce_loss = self.ce_loss_fct(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))

        # Teacher CE over CoT + answer.
        if ref_outputs is not None:
            ref_logits = ref_outputs.logits
            ref_ce_loss = self.ce_loss_fct(
                ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1)), ref_labels[:, 1:].reshape(-1)
            )
            ref_ce_loss = ref_ce_loss * self.teacher_weight
        else:
            ref_ce_loss = torch.tensor(0.0, device=encoder_input_ids.device)

        distill_loss = distill_loss * self.distill_weight
        loss = ce_loss + distill_loss + ref_ce_loss

        self._stored_metrics["train"]["student_ce_loss"].append(ce_loss.detach().item())
        self._stored_metrics["train"]["distill_loss"].append(distill_loss.detach().item())
        self._stored_metrics["train"]["teacher_ce_loss"].append(ref_ce_loss.detach().item())
        return loss

    def _save_extras(self, output_dir: str) -> None:
        """Save the projection pi (needed at inference) and the run config next to the model weights."""
        if self.projection is not None:
            import deepspeed
            from safetensors.torch import save_file

            with deepspeed.zero.GatheredParameters(list(self.projection.parameters()), modifier_rank=0):
                if self.args.local_process_index == 0:
                    os.makedirs(output_dir, exist_ok=True)
                    state = detached_state_dict(self.projection.state_dict())
                    save_file(state, os.path.join(output_dir, PROJECTION_WEIGHTS_NAME))

        if self.is_world_process_zero():
            save_run_config(output_dir, self.run_config)

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        self._save_extras(output_dir if output_dir is not None else self.args.output_dir)

    @override
    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        self._save_extras(os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}"))
