"""Configuration of OpenTSLM-Flamingo: a Llama LM with a CNN tokenizer, a perceiver and gated cross-attention."""

from transformers.models.llama.configuration_llama import LlamaConfig


class OpenTSLMFlamingoConfig(LlamaConfig):
    model_type = "opentslm_flamingo"

    def __init__(
        self,
        vis_dim: int = 128,  # width of the CNN tokenizer, perceiver and cross-attention
        ts_patch_size: int = 4,  # CNN tokenizer kernel and stride
        cross_attn_every_n_layers: int = 1,
        perceiver_num_latents: int = 64,
        perceiver_depth: int = 6,
        max_patches: int = 1024,
        media_token_id: int | None = None,  # id of "<image>"
        eoc_token_id: int | None = None,  # id of "<|endofchunk|>"
        normalize_timeseries: bool = True,
        ignore_index: int = -100,
        **kwargs,
    ):
        self.vis_dim = vis_dim
        self.ts_patch_size = ts_patch_size
        self.cross_attn_every_n_layers = cross_attn_every_n_layers
        self.perceiver_num_latents = perceiver_num_latents
        self.perceiver_depth = perceiver_depth
        self.max_patches = max_patches
        self.media_token_id = media_token_id
        self.eoc_token_id = eoc_token_id
        self.normalize_timeseries = normalize_timeseries
        self.ignore_index = ignore_index
        super().__init__(**kwargs)
