"""SLIP (a Gemma3 or Llama LM that cross-attends to a sensor encoder) as a Hugging Face causal LM."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput

from .configuration_slip import SlipConfig
from .multimodal_gemma import Gemma3MultimodalModel, slip_vis_x_ctx
from .multimodal_llama import LlamaMultimodalModel
from .ts_transformer import AttentionPooling, SensorTransformerModel


@dataclass
class SlipCausalLMOutput(ModelOutput):
    """Causal LM outputs; ``attention_mask`` is always None since the sequence is never expanded."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[tuple] = None
    attention_mask: Optional[torch.Tensor] = None


class SlipForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SlipConfig
    supports_gradient_checkpointing = True
    _supports_cache_class = True

    def __init__(self, config: SlipConfig):
        super().__init__(config)

        if config.sensor_encoder is None:
            raise ValueError(
                "SlipConfig.sensor_encoder is required; pass the dict captured from the "
                "original hydra config (embed_dim, num_heads, mlp_ratio, depth, dropout_rate, ...)."
            )

        self.sensor_encoder = SensorTransformerModel(**config.sensor_encoder)

        if "llama" in config.llm_model_name.lower():
            backbone_cls = LlamaMultimodalModel
        else:
            backbone_cls = Gemma3MultimodalModel

        self.multimodalModel = backbone_cls(
            model_id=config.llm_model_name,
            post_train=config.post_train,
            split_layer=config.split_layer,
            num_heads=config.num_heads,
        )

        # Match the vocabulary of checkpoints trained with added special tokens.
        inner_vocab = self.multimodalModel.model.config.vocab_size
        if config.vocab_size != inner_vocab:
            # mean_resizing would run on meta tensors; the checkpoint overwrites the new rows anyway.
            self.multimodalModel.model.resize_token_embeddings(
                config.vocab_size, mean_resizing=False
            )
            self.multimodalModel.model.config.vocab_size = config.vocab_size

        # The generation cache reads the decoder's attributes off self.config.
        inner_cfg = self.multimodalModel.model.config
        for attr in (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "rope_scaling",
            "sliding_window",
            "sliding_window_pattern",
            "layer_types",
            "attention_bias",
            "attention_dropout",
            "hidden_activation",
            "tie_word_embeddings",
            "attn_implementation",
        ):
            if hasattr(inner_cfg, attr):
                setattr(self.config, attr, getattr(inner_cfg, attr))

        sensor_dim = self.sensor_encoder.embed_dim
        lm_dim = self.multimodalModel.hidden_size

        if config.num_img_queries > 0:
            self.img_queries = nn.Parameter(torch.randn(config.num_img_queries + 1, lm_dim))
            pool_heads = config.img_attn_pool_num_heads or config.num_heads
            self.img_attn_pool = AttentionPooling(
                dim=lm_dim, context_dim=sensor_dim, num_heads=pool_heads
            )
        else:
            if sensor_dim != lm_dim:
                raise ValueError(
                    f"sensor_encoder.embed_dim ({sensor_dim}) must equal llm hidden_size ({lm_dim}) "
                    "when num_img_queries == 0, or enable pooling by setting num_img_queries > 0."
                )

        self._cached_sensor_hidden: Optional[torch.Tensor] = None

    def get_input_embeddings(self):
        return self.multimodalModel.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.multimodalModel.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.multimodalModel.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.multimodalModel.model.set_output_embeddings(new_embeddings)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.multimodalModel.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.multimodalModel.model.gradient_checkpointing_disable()

    def _embed_sensor(
        self,
        timeseries,
        sensor_attn_mask,
        time_index,
    ) -> torch.Tensor:
        sensor_tokens, attn_mask = self.sensor_encoder(timeseries, sensor_attn_mask, time_index=time_index)

        if hasattr(self, "img_attn_pool"):
            batch = sensor_tokens.shape[0]
            img_queries = repeat(self.img_queries, "n d -> b n d", b=batch)
            sensor_tokens = self.img_attn_pool(img_queries, sensor_tokens, attn_mask)

        return sensor_tokens

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        timeseries=None,
        sensor_attn_mask=None,
        time_index=None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ts_patch_sizes=None,  # unused (ChatTS interface)
        labels=None,
        **kwargs,
    ):
        if timeseries is not None:
            sensor_hidden = self._embed_sensor(timeseries, sensor_attn_mask, time_index)
            self._cached_sensor_hidden = sensor_hidden
        else:
            sensor_hidden = self._cached_sensor_hidden
            if sensor_hidden is None:
                raise RuntimeError(
                    "SlipForCausalLM.forward called without `timeseries` and no sensor "
                    "features are cached. Pass sensors on the encoder pass first."
                )

        with slip_vis_x_ctx(sensor_hidden):
            base_outputs = self.multimodalModel.model.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        last_hidden_state = base_outputs.last_hidden_state
        # ``get_output_embeddings`` is stable across PEFT wrapping; ``lm_head`` isn't.
        logits = self.get_output_embeddings()(last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return SlipCausalLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=last_hidden_state,
            hidden_states=getattr(base_outputs, "hidden_states", None),
            past_key_values=getattr(base_outputs, "past_key_values", None),
            attention_mask=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        timeseries=None,
        sensor_attn_mask=None,
        time_index=None,
        **kwargs,
    ):
        # Prefill vs. decode by cache length: the first step may receive an empty cache.
        cache_len = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                cache_len = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
                cache_len = past_key_values[0][0].shape[2]

        if cache_len > 0:
            input_ids = input_ids[:, -1:]
            timeseries = None
            sensor_attn_mask = None
            time_index = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "timeseries": timeseries,
            "sensor_attn_mask": sensor_attn_mask,
            "time_index": time_index,
            "use_cache": kwargs.get("use_cache", True),
        }
