"""OpenTSLM-Flamingo as a Hugging Face causal LM with the interface of the SLIP backbone."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.utils import ModelOutput, logging

from .configuration_opentslm_flamingo import OpenTSLMFlamingoConfig

from open_flamingo.src.flamingo_lm import FlamingoLMMixin, FlamingoLayer
from open_flamingo.src.helpers import PerceiverResampler
from open_flamingo.src.utils import extend_instance

logger = logging.get_logger(__name__)


# Recent transformers read `attention_type` off every decoder layer, which FlamingoLayer lacks.
if not isinstance(getattr(FlamingoLayer, "attention_type", None), property):
    def _attention_type_property(self):
        return getattr(self.decoder_layer, "attention_type", None)

    FlamingoLayer.attention_type = property(_attention_type_property)  # type: ignore


PATCH_SIZE = 4
ENCODER_OUTPUT_DIM = 128


class _CNNTokenizer(nn.Module):
    """OpenTSLM's CNN tokenizer: ``(B, L)`` series -> ``(B, L / patch_size, D)`` patch embeddings."""

    def __init__(
        self,
        transformer_input_dim: int = ENCODER_OUTPUT_DIM,
        patch_size: int = PATCH_SIZE,
        max_patches: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, transformer_input_dim))
        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(f"Sequence length {L} not divisible by patch_size {self.patch_size}")
        x = x.unsqueeze(1)             # [B, 1, L]
        x = self.patch_embed(x)        # [B, D, N]
        x = x.transpose(1, 2)          # [B, N, D]
        N = x.size(1)
        if N > self.pos_embed.size(1):
            raise ValueError(
                f"Series of {N * self.patch_size} steps too long; max "
                f"{self.pos_embed.size(1) * self.patch_size} (raise max_patches)."
            )
        x = x + self.pos_embed[:, :N, :]
        x = self.input_norm(x)
        x = self.input_dropout(x)
        return x                       # [B, N, D]


@dataclass
class OpenTSLMFlamingoCausalLMOutput(ModelOutput):
    """Causal LM outputs; ``attention_mask`` is always None since the sequence is never expanded."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple] = None
    past_key_values: Optional[tuple] = None
    attention_mask: Optional[torch.Tensor] = None


def _sanitized_llama_config(config: OpenTSLMFlamingoConfig) -> LlamaConfig:
    """Llama config of the inner language model, without the Flamingo fields."""
    drop = {
        "vis_dim", "ts_patch_size", "cross_attn_every_n_layers", "perceiver_num_latents",
        "perceiver_depth", "max_patches", "media_token_id", "eoc_token_id",
        "normalize_timeseries", "ignore_index",
        "model_type", "architectures", "auto_map", "_name_or_path", "torch_dtype",
    }
    d = {k: v for k, v in config.to_dict().items() if k not in drop}
    llama_cfg = LlamaConfig(**d)
    # The published weights were trained with eager attention.
    llama_cfg._attn_implementation = "eager"
    return llama_cfg


class OpenTSLMFlamingoForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = OpenTSLMFlamingoConfig
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _no_split_modules = ["LlamaDecoderLayer", "FlamingoLayer"]

    def __init__(self, config: OpenTSLMFlamingoConfig):
        super().__init__(config)

        self.vision_encoder = _CNNTokenizer(
            transformer_input_dim=config.vis_dim,
            patch_size=config.ts_patch_size,
            max_patches=config.max_patches,
        )

        # Llama with open_flamingo's gated cross-attention layers.
        lang_encoder = LlamaForCausalLM(_sanitized_llama_config(config))
        extend_instance(lang_encoder, FlamingoLMMixin)
        lang_encoder.set_decoder_layers_attr_name("model.layers")
        lang_encoder.init_flamingo(
            media_token_id=(config.media_token_id if config.media_token_id is not None else 0),
            lang_hidden_size=config.hidden_size,
            vis_hidden_size=config.vis_dim,
            cross_attn_every_n_layers=config.cross_attn_every_n_layers,
            gradient_checkpointing=False,
        )
        self.lang_encoder = lang_encoder

        self.perceiver = PerceiverResampler(
            dim=config.vis_dim,
            depth=config.perceiver_depth,
            num_latents=config.perceiver_num_latents,
        )
        self.perceiver._use_gradient_checkpointing = False

        self.media_token_id = config.media_token_id
        self.eoc_token_id = config.eoc_token_id
        self.vis_dim = config.vis_dim
        self._cached_vis_x: Optional[torch.Tensor] = None

        self.post_init()

    @property
    def sensor_encoder(self) -> nn.Module:
        """Alias of the CNN tokenizer, not registered as a second submodule."""
        return self.vision_encoder

    @property
    def embed_dim(self) -> int:
        return self.vis_dim

    def get_input_embeddings(self):
        return self.lang_encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.lang_encoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lang_encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lang_encoder.set_output_embeddings(new_embeddings)

    def tie_weights(self):
        return self.lang_encoder.tie_weights()

    def resize_token_embeddings(self, new_num_tokens=None, *args, **kwargs):
        out = self.lang_encoder.resize_token_embeddings(new_num_tokens, *args, **kwargs)
        if new_num_tokens is not None:
            actual_vocab = out.weight.shape[0]  # includes the pad_to_multiple_of padding
            self.config.vocab_size = actual_vocab
            self.lang_encoder.config.vocab_size = actual_vocab
        return out

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lang_encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.lang_encoder.model.gradient_checkpointing_disable()

    def _encode_vision_x(self, vision_x: torch.Tensor) -> torch.Tensor:
        """``(B, T_img, F, L)`` series -> perceiver latents ``(B, T_img, num_latents, vis_dim)``."""
        w = self.vision_encoder.patch_embed.weight
        vision_x = vision_x.to(device=w.device, dtype=w.dtype)
        if vision_x.ndim != 4:
            raise ValueError(
                f"Flamingo vision_x must be (B, T_img, F, L); got shape {tuple(vision_x.shape)}"
            )
        b, T, F, L = vision_x.shape
        x = vision_x.reshape(b * T * F, L)                 # (b*T*F, L)
        x = self.vision_encoder(x)                          # (b*T*F, N, D)
        x = x.reshape(b, T, F, x.size(1), x.size(2))        # (b, T, F, N, D)
        x = self.perceiver(x)                               # (b, T, num_latents, D)
        return x

    def _set_conditioning(self, vis_x: torch.Tensor, input_ids: Optional[torch.LongTensor]) -> None:
        """Condition the cross-attention layers; inputs without <image> markers reuse the cached media."""
        layers = self.lang_encoder._get_decoder_layers()
        media_locations = None
        if input_ids is not None:
            media_locations = input_ids == self.media_token_id

        use_cached = bool(
            getattr(self.lang_encoder, "_use_cached_vision_x", False)
            and (media_locations is None or not media_locations.any())
        )
        for layer in layers:
            layer.condition_vis_x(vis_x)
            if not use_cached:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        timeseries: Optional[torch.FloatTensor] = None,
        sensor_attn_mask=None,        # used by sensor masking, not by the model
        time_index=None,              # unused (SLIP interface)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        ts_patch_sizes=None,          # unused (ChatTS interface)
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> OpenTSLMFlamingoCausalLMOutput:
        # Encode the series when given; later forwards without it reuse this encoding.
        if timeseries is not None:
            vis_x = self._encode_vision_x(timeseries)
            self._cached_vis_x = vis_x
            self.lang_encoder._use_cached_vision_x = True
        else:
            vis_x = self._cached_vis_x
            if vis_x is None:
                raise RuntimeError(
                    "OpenTSLMFlamingoForCausalLM.forward called without `timeseries` and no "
                    "vision features are cached. Pass the time-series on the first (prefill) pass."
                )

        self._set_conditioning(vis_x, input_ids)
        base_outputs = self.lang_encoder.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
        )
        last_hidden_state = base_outputs.last_hidden_state
        logits = self.lang_encoder.lm_head(last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.ignore_index,
            )

        return OpenTSLMFlamingoCausalLMOutput(
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
        cache_position=None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Prefill vs. decode by cache length: the first step may receive an empty cache.
        cache_len = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                cache_len = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (tuple, list)) and past_key_values and past_key_values[0] is not None:
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
            "cache_position": cache_position,
            "use_cache": kwargs.get("use_cache", True),
        }


__all__ = ["OpenTSLMFlamingoForCausalLM", "OpenTSLMFlamingoConfig"]
