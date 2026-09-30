"""Llama decoder with SLIP cross-attention blocks from ``split_layer`` on."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .multimodal_gemma import Residual, _SLIP_VIS_X
from .ts_transformer import CrossAttention


class LlamaMultimodalLayer(nn.Module):
    """Llama decoder layer followed by cross-attention to the sensor context bound by ``slip_vis_x_ctx``."""

    def __init__(self, original_layer, cross_attn_block):
        super().__init__()
        self.original_layer = original_layer
        self.cross_attn_block = cross_attn_block

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original_layer = super().__getattr__("original_layer")
            except AttributeError:
                raise AttributeError(name)
            return getattr(original_layer, name)

    def forward(self, hidden_states, **kwargs):
        vis_x = _SLIP_VIS_X.get()
        assert vis_x is not None, (
            "SLIP cross-attention invoked without a sensor context. "
            "Wrap the model forward in `slip_vis_x_ctx(sensor_hidden)` first."
        )

        outputs = self.original_layer(hidden_states, **kwargs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        vis_x_cast = vis_x.to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = self.cross_attn_block(hidden_states, context=vis_x_cast)

        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]
        return hidden_states


class LlamaMultimodalModel(nn.Module):
    """Standard HF Llama with SLIP cross-attention inserted at/after ``split_layer``."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-1B",
        post_train: bool = False,
        split_layer: int = 12,
        num_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        load_kwargs = dict(attn_implementation="sdpa", trust_remote_code=True)
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        # The SLIP checkpoint holds the full LM weights, so by default only the config is loaded here.
        if post_train:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        else:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_config(config, **load_kwargs)

        self.split_layer = split_layer
        hidden_size = self.model.config.hidden_size
        self.hidden_size = hidden_size
        if num_heads is None:
            num_heads = self.model.config.num_attention_heads

        for i in range(split_layer, len(self.model.model.layers)):
            cross_attn = CrossAttention(
                dim=hidden_size, context_dim=hidden_size, num_heads=num_heads, dropout_rate=0.1
            )
            original_layer = self.model.model.layers[i]
            self.model.model.layers[i] = LlamaMultimodalLayer(original_layer, Residual(cross_attn))
