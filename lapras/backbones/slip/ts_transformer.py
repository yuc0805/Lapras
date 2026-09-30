"""SensorTransformerModel — the SLIP time-series encoder.

Adapted from HealthSensorSLM-Bench/model_factory/ts_transformer.py for use as
a standalone module in the SLIP Hub repo. Changes from the original:

- Relative import of RotaryEmbedding / apply_rotary_pos_emb from ``.pos_embed``.
- Removed the ``AllAttention`` branch (and its ``TsRoPEAttention`` dependency);
  only the default ``group_attn`` and ``univariate`` channel attention paths
  are supported, which is what the current SLIP checkpoint uses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange, reduce
from transformers.activations import ACT2FN


# Rotary positional embedding helpers.

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=10000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def apply_rotary_pos_emb_2d(q, k, cos_h, sin_h, cos_w, sin_w, pos_h, pos_w, unsqueeze_dim=1):
    Dh = q.shape[-1]
    q_h, q_w = q.split(Dh // 2, dim=-1)
    k_h, k_w = k.split(Dh // 2, dim=-1)
    q_h, k_h = apply_rotary_pos_emb(q_h, k_h, cos_h, sin_h, pos_h.long(), unsqueeze_dim=unsqueeze_dim)
    q_w, k_w = apply_rotary_pos_emb(q_w, k_w, cos_w, sin_w, pos_w.long(), unsqueeze_dim=unsqueeze_dim)
    return torch.cat([q_h, q_w], dim=-1), torch.cat([k_h, k_w], dim=-1)


def build_2d_position_ids(attention_mask, flatten=True):
    B, V, P = attention_mask.shape
    mask = attention_mask.to(dtype=torch.long)
    pos_patch = (mask.cumsum(dim=-1) - 1) * mask
    var_valid = mask.any(dim=-1).to(dtype=torch.long)
    pos_var_base = (var_valid.cumsum(dim=1) - 1) * var_valid
    pos_var = pos_var_base.unsqueeze(-1).expand(B, V, P) * mask
    if flatten:
        return pos_var.reshape(B, V * P).long(), pos_patch.reshape(B, V * P).long()
    return pos_var.long(), pos_patch.long()


def flatten_list(input_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    return [item for sublist in input_list for item in sublist]


class MultiSizePatchEmbed(nn.Module):
    def __init__(self, base_patch=32, **cfg):
        super().__init__()
        self.base_patch = base_patch
        hidden_size = cfg["embed_dim"]
        intermediate_size = cfg["mlp_ratio"] * hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        self.shared_linear = nn.Linear(base_patch * 3, intermediate_size)
        self.shared_residual = nn.Linear(base_patch * 3, hidden_size)

        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.act = ACT2FN["silu"]
        self.output_layer = nn.Linear(intermediate_size, hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def resize_weight(self, patch_size: int):
        base_w = self.shared_linear.weight
        base_b = self.shared_linear.bias
        res_w = self.shared_residual.weight
        res_b = self.shared_residual.bias

        new_w = F.interpolate(
            base_w.unsqueeze(1), size=patch_size, mode="linear", align_corners=False
        ).squeeze(1).to(base_w.dtype)
        new_res_w = F.interpolate(
            res_w.unsqueeze(1), size=patch_size, mode="linear", align_corners=False
        ).squeeze(1).to(res_w.dtype)
        return new_w, base_b, new_res_w, res_b

    def forward(self, x_list, attention_mask, time_idx):
        param_dtype = self.shared_linear.weight.dtype
        device = self.shared_linear.weight.device

        sizes = torch.tensor([x.shape[-1] for x in x_list])
        unique_sizes = sizes.unique(sorted=True)
        N = x_list[0].shape[0]

        outputs = torch.empty(len(x_list), N, self.intermediate_size, device=device, dtype=param_dtype)
        res_outputs = torch.empty(len(x_list), N, self.hidden_size, device=device, dtype=param_dtype)

        for psize in unique_sizes.tolist():
            idxs = (sizes == psize).nonzero(as_tuple=True)[0]
            xs = torch.stack([x_list[i] for i in idxs])
            mask = torch.stack([attention_mask[i] for i in idxs])
            ti = torch.stack([time_idx[i] for i in idxs])

            xs = xs.to(device=device, dtype=param_dtype, non_blocking=True)
            mask = mask.to(device=device, dtype=param_dtype, non_blocking=True)
            ti = ti.to(device=device, dtype=param_dtype, non_blocking=True)

            xs = torch.cat([xs, mask, ti], dim=-1)
            w, b, r_w, r_b = self.resize_weight(psize * 3)

            res_outputs[idxs] = F.linear(xs, r_w, r_b)
            outputs[idxs] = F.linear(xs, w, b)

        hid = self.act(outputs)
        out = self.dropout(self.output_layer(hid))
        out = out + res_outputs
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        patch_size = cfg["patch_size"]
        self.patch_size = patch_size

        self.dropout = nn.Dropout(cfg.get("dropout_rate", 0.1))
        hidden_size = cfg["embed_dim"]
        intermediate_size = hidden_size * 4

        self.hidden_layer = nn.Linear(patch_size * 3, intermediate_size)
        self.act = ACT2FN["silu"]
        self.output_layer = nn.Linear(intermediate_size, hidden_size)
        self.residual_layer = nn.Linear(patch_size * 3, hidden_size)

    def forward(self, x, mask, time_idx):
        x = rearrange(x, "bs nvar (nump ps) -> (bs nvar) nump ps", ps=self.patch_size)
        mask = rearrange(mask, "bs nvar (nump ps) -> (bs nvar) nump ps", ps=self.patch_size)
        time_idx = rearrange(time_idx, "bs nvar (nump ps) -> (bs nvar) nump ps", ps=self.patch_size)

        x = torch.cat([x, mask, time_idx], dim=-1)
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        return out + res


class Attention(nn.Module):
    def __init__(self, layer_idx: int, is_rope=True, **cfg):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_rope = is_rope
        self.hidden_size = cfg.get("embed_dim", 768)
        self.num_heads = cfg.get("num_heads", 12)
        self.sensor_max_len = cfg.get("sensor_max_len", 2880)
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = cfg.get("dropout_rate", 0.1)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.is_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.sensor_max_len)
        else:
            self.rotary_emb = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.is_rope:
            kv_seq_len = key_states.shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class TsRoPEAttention(nn.Module):
    """2D-RoPE self-attention used by the ``all_attn`` channel path."""

    def __init__(self, layer_idx: int, **cfg):
        super().__init__()
        self.hidden_size = cfg.get("embed_dim", 768)
        self.num_heads = cfg.get("num_heads", 12)
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = cfg.get("dropout_rate", 0.1)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim // 2, max_position_embeddings=cfg.get("max_position_embeddings")
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        tmp_attn_mask = rearrange(attention_mask, "b nvar p -> b (nvar p)")
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        tmp_attn_mask = tmp_attn_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, q_len, q_len).bool()
        pos_var, pos_patch = build_2d_position_ids(attention_mask, flatten=True)
        cos_h, sin_h = self.rotary_emb(query_states, seq_len=int(pos_var.max().item()) + 1)
        cos_w, sin_w = self.rotary_emb(query_states, seq_len=int(pos_patch.max().item()) + 1)
        query_states, key_states = apply_rotary_pos_emb_2d(
            query_states, key_states, cos_h, sin_h, cos_w, sin_w, pos_var, pos_patch
        )
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, tmp_attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class AllAttention(nn.Module):
    def __init__(self, layer_idx: int, **cfg):
        super().__init__()
        self.self_attention = TsRoPEAttention(layer_idx=layer_idx, **cfg)
        self.layer_norm = nn.LayerNorm(cfg.get("embed_dim"))
        self.dropout = nn.Dropout(cfg.get("dropout_rate", 0.1))

    def forward(self, hidden_states, attention_mask):
        return hidden_states + self.dropout(
            self.self_attention(self.layer_norm(hidden_states), attention_mask)
        )


class CrossAttention(nn.Module):
    def __init__(self, dim=768, *, context_dim=384, num_heads=12, dropout_rate=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = int(dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = dropout_rate

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(context_dim, dim, bias=True)
        self.v_proj = nn.Linear(context_dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query, context, attention_mask=None, **kwargs):
        bsz, q_len, _ = query.size()
        bsc, k_len, _ = context.size()
        assert bsz == bsc, f"Batch size mismatch: {bsz} vs {bsc}"

        query = self.norm(query)
        context = self.context_norm(context)

        query_states = self.q_proj(query)
        key_states = self.k_proj(context)
        value_states = self.v_proj(context)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.dim)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class TimeSelfAttention(nn.Module):
    def __init__(self, layer_idx, **cfg):
        super().__init__()
        self.self_attention = Attention(layer_idx=layer_idx, is_rope=True, **cfg)
        self.layer_norm = nn.LayerNorm(cfg.get("embed_dim", 768))
        self.dropout = nn.Dropout(cfg.get("dropout_rate", 0.1))

    def forward(self, hidden_states, attention_mask, position_ids):
        q_len = hidden_states.size(1)
        attention_mask = rearrange(attention_mask, "b nvar p -> (b nvar) p")
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, q_len, q_len).bool()

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.self_attention(normed_hidden_states, attention_mask, position_ids)
        return hidden_states + self.dropout(attention_output)


class GroupSelfAttention(nn.Module):
    def __init__(self, layer_idx: int, **cfg):
        super().__init__()
        self.self_attention = Attention(layer_idx, is_rope=False, **cfg)
        self.layer_norm = nn.LayerNorm(cfg.get("embed_dim", 768))
        self.dropout = nn.Dropout(cfg.get("dropout_rate", 0.1))

    def forward(self, hidden_states, attention_mask, group_ids):
        BS, nvar, _ = attention_mask.shape
        hidden_states = rearrange(hidden_states, "(bs nvar) l d -> (bs l) nvar d", bs=BS, nvar=nvar)
        attention_mask = rearrange(attention_mask, "bs nvar l -> (bs l) nvar")
        group_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, nvar, nvar).bool()

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.self_attention(normed_hidden_states, group_attn_mask)
        hidden_states = hidden_states + self.dropout(attention_output)
        return rearrange(hidden_states, "(bs l) nvar d -> (bs nvar) l d", bs=BS, nvar=nvar)


class AttentionPooling(nn.Module):
    def __init__(self, dim=768, mlp_ratio=4, context_dim=384, num_heads=12, dropout_rate=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(
            dim=dim, context_dim=context_dim, num_heads=num_heads, dropout_rate=dropout_rate
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_layer = MLP(hidden_size=dim, intermediate_size=dim * mlp_ratio, hidden_act="silu")
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, context, attn_mask=None):
        b, n, _ = x.shape
        kv_len = context.shape[1]

        attn_mask = rearrange(attn_mask, "b nvar p -> b (nvar p)")
        attn_mask = attn_mask.view(b, 1, 1, kv_len).expand(b, 1, n, kv_len).bool()

        x = self.cross_attn(x, context, attn_mask)
        x = x + self.ffn_layer(self.ffn_norm(x))
        return self.post_norm(x)


class SensorEncoderLayer(nn.Module):
    def __init__(self, layer_idx: int, **cfg):
        super().__init__()
        hidden_size = cfg["embed_dim"]
        intermediate_size = cfg["mlp_ratio"] * hidden_size

        self.channel_attn_type = cfg.get("channel_attn_type", "group_attn")
        if self.channel_attn_type == "group_attn":
            self.ts_attn = TimeSelfAttention(layer_idx=layer_idx, **cfg)
            self.group_attn = GroupSelfAttention(layer_idx=layer_idx, **cfg)
        elif self.channel_attn_type == "univariate":
            self.ts_attn = TimeSelfAttention(layer_idx=layer_idx, **cfg)
        elif self.channel_attn_type == "all_attn":
            self.ts_attn = AllAttention(layer_idx=layer_idx, **cfg)
        else:
            raise ValueError(f"Unsupported channel_attn_type={self.channel_attn_type!r}")

        self.norm = nn.LayerNorm(hidden_size)
        self.ffn_layer = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act="silu")

    def forward(self, hidden_states, attention_mask=None, group_ids=None, position_ids=None):
        if self.channel_attn_type == "group_attn":
            hidden_states = self.ts_attn(hidden_states, attention_mask, position_ids)
            hidden_states = self.group_attn(hidden_states, attention_mask, group_ids)
        elif self.channel_attn_type == "all_attn":
            hidden_states = self.ts_attn(hidden_states, attention_mask)
        else:
            hidden_states = self.ts_attn(hidden_states, attention_mask, position_ids)

        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.ffn_layer(hidden_states)
        return residual + hidden_states


class SensorTransformerModel(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        patch_size = cfg.get("patch_size", None)
        self.patch_size = patch_size
        if patch_size is not None:
            self.patch_embed = PatchEmbedding(**cfg)
        else:
            self.patch_embed = MultiSizePatchEmbed(**cfg)

        self.blocks = nn.ModuleList([SensorEncoderLayer(layer_idx, **cfg) for layer_idx in range(cfg["depth"])])
        self.norm = torch.nn.LayerNorm(cfg["embed_dim"])
        self.embed_dim = cfg["embed_dim"]
        self.channel_attn_type = cfg.get("channel_attn_type", "group_attn")

    def forward(self, input_ids, attention_mask, time_index):
        if self.patch_size is None:
            BS = len(input_ids)
            flat_input_ids = flatten_list(input_ids)
            flat_attention_mask = flatten_list(attention_mask)
            flat_time_index = flatten_list(time_index)

            hidden_states = self.patch_embed(flat_input_ids, flat_attention_mask, flat_time_index)
            attention_mask = self._get_self_attn_mask(attention_mask).to(hidden_states.device)
            position_ids = self._build_rope_position_ids(attention_mask)
            position_ids = rearrange(position_ids, "b nvar p -> (b nvar) p")
        else:
            BS, nvar, L = input_ids.shape
            hidden_states = self.patch_embed(input_ids, attention_mask, time_index)
            attention_mask = reduce(
                attention_mask, "b v (p ps) -> b v p", "max", ps=self.patch_size
            )
            position_ids = self._build_rope_position_ids(attention_mask)
            position_ids = rearrange(position_ids, "b nvar p -> (b nvar) p")

        if self.channel_attn_type == "all_attn":
            hidden_states = rearrange(hidden_states, "(b nvar) l d -> b (nvar l) d", b=BS)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, attention_mask=attention_mask, group_ids=None, position_ids=position_ids)

        if self.channel_attn_type == "group_attn":
            hidden_states = rearrange(hidden_states, "(b nvar) l d -> b (nvar l) d", b=BS)

        hidden_states = self.norm(hidden_states)
        return hidden_states, attention_mask

    def _build_rope_position_ids(self, attention_mask):
        assert attention_mask.dim() == 3
        mask = attention_mask.to(torch.long)
        pos = (mask.cumsum(dim=-1) - 1) * mask
        return pos

    def _get_self_attn_mask(self, attn_mask_list):
        collapsed_batch = []
        for sample_masks in attn_mask_list:
            nvar_collapsed = [
                (var_mask.sum(dim=-1) > 0).to(var_mask.dtype) for var_mask in sample_masks
            ]
            nvar_collapsed = torch.stack(nvar_collapsed, dim=0)
            collapsed_batch.append(nvar_collapsed)
        return torch.stack(collapsed_batch, dim=0)
