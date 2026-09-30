"""Configuration of SlipForCausalLM."""

from typing import Any, Optional

from transformers import PretrainedConfig


class SlipConfig(PretrainedConfig):
    """Config of SLIP; ``sensor_encoder`` holds the sensor-encoder kwargs as a plain dict."""

    model_type = "slip"

    def __init__(
        self,
        llm_model_name: str = "google/gemma-3-270m",
        split_layer: int = 12,
        sensor_encoder: Optional[dict[str, Any]] = None,
        num_img_queries: int = 0,
        num_heads: int = 5,
        img_attn_pool_num_heads: Optional[int] = None,  # None: num_heads
        hidden_size: int = 640,
        post_train: bool = False,
        max_llm_len: int = 768,
        vocab_size: int = 262144,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        **kwargs,
    ):
        self.llm_model_name = llm_model_name
        self.split_layer = split_layer
        self.sensor_encoder = dict(sensor_encoder) if sensor_encoder is not None else None
        self.num_img_queries = num_img_queries
        self.num_heads = num_heads
        self.img_attn_pool_num_heads = img_attn_pool_num_heads
        self.hidden_size = hidden_size
        self.post_train = post_train
        self.max_llm_len = max_llm_len
        self.vocab_size = vocab_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
