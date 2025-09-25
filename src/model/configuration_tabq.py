# -*- coding:utf-8 -*-
# @FileName : configuration_tabq.py
# @Time : 2025/02/28 23:50
# @Author : fiv

from transformers import Qwen2Config, GotOcr2Config


class TabQConfig(Qwen2Config):
    model_type = "tabq"
    sub_configs = {"vision_config": GotOcr2Config}

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        vision_config=None,
        model_part="llm,vis",
        just_got_image_features=False,
        image_token_id=151859,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, GotOcr2Config):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        self.model_part = model_part
        self.just_got_image_features = just_got_image_features
        self.image_token_id = image_token_id

        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            attention_dropout,
            **kwargs,
        )
