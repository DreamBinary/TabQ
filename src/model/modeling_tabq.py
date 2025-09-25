# -*- coding:utf-8 -*-
# @FileName : tabq.py
# @Time : 2025/02/28 22:51
# @Author : fiv

import os
import torch
import torch.nn as nn
from safetensors import safe_open
from typing import Dict, List, Optional, Tuple, Union, Type
from transformers import (
    PreTrainedModel,
    Qwen2ForCausalLM,
    GotOcr2ForConditionalGeneration,
)
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import SpecificPreTrainedModelType
from .configuration_tabq import TabQConfig


class ImageFeaturePicker(nn.Module):
    """
    Image Feature Picker. A pooling module for the image features.
    This module selects the image features from the output of the GotOcr2 model.
    """

    def __init__(self, num_image_tokens, input_dim, hidden_size, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        self.queries = nn.Parameter(torch.randn(1, num_image_tokens, hidden_size))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        x = self.proj(
            x
        )  # from [B, N, I] to [B, N, H] : [B, 296, 1024] -> [B, 296, 1536]
        q = self.queries.expand(x.size(0), -1, -1)  # [B, T, H] : [B, 256, 1536]
        hidden_state, _ = self.attn(
            q, x, x
        )  # from [B, N, H] to [B, T, H] : [B, 296, 1536] -> [B, 256, 1536]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = hidden_state + residual

        return hidden_state


class TabQModel(PreTrainedModel, GenerationMixin):
    """
    Tabular Quest Model for tabular data
    """

    config_class = TabQConfig

    def __init__(self, config, ocr_config=None):
        super().__init__(config)
        self.config = config
        if "llm" in config.model_part:
            self.llm = Qwen2ForCausalLM(config)
            self.just_got_image_features = config.just_got_image_features
            self.image_feature_picker = ImageFeaturePicker(
                ocr_config["num_image_tokens"],
                config.vision_config.text_config.hidden_size,
                self.config.hidden_size,
            )
            if self.llm._tied_weights_keys is not None:
                self._tied_weights_keys = [
                    f"llm.{k}" for k in self.llm._tied_weights_keys
                ]
        if "vis" in config.model_part:
            self.vis = GotOcr2ForConditionalGeneration(config.vision_config)

    def load_weight(self, llm_model_name=None, vis_model_name=None, ocr_config=None):
        if vis_model_name is not None:
            self.vis = GotOcr2ForConditionalGeneration.from_pretrained(vis_model_name)
            self.config.vision_config = self.vis.config

        if llm_model_name is not None:
            self.llm = Qwen2ForCausalLM.from_pretrained(llm_model_name)
            llm_config = self.llm.config
            for k, v in llm_config.to_diff_dict().items():
                if k not in ["architectures", "model_type", "_name_or_path"]:
                    self.config.__dict__[k] = v
            self.just_got_image_features = self.config.just_got_image_features
            self.image_feature_picker = ImageFeaturePicker(
                ocr_config["num_image_tokens"],
                self.config.vision_config.text_config.hidden_size,
                self.config.hidden_size,
            )
            if self.llm._tied_weights_keys is not None:
                self._tied_weights_keys = [
                    f"llm.{k}" for k in self.llm._tied_weights_keys
                ]
        self.config.model_part = ",".join(
            part for part in ["llm", "vis"] if hasattr(self, part)
        )

    def load_weight_from_safetensors(self, safetensors_dir):
        state_dict = {}
        for file in os.listdir(safetensors_dir):
            if file.endswith(".safetensors"):
                with safe_open(
                    os.path.join(safetensors_dir, file), framework="pt"
                ) as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        self.load_state_dict(state_dict, strict=False)

    def _ocr(self, vis_inputs, just_image_features=False):
        if just_image_features:
            image_features = self.vis.get_image_features(
                pixel_values=vis_inputs["pixel_values"]
            )
            return image_features

        output = self.vis(
            **vis_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        return output

    def _merge_image_features(self, inputs_embeds, input_ids, vis_inputs):
        if self.just_got_image_features:
            image_features = self._ocr(
                vis_inputs, just_image_features=True
            )  # [1, 256, 1024] -> [B, T, H]
        else:
            hidden_states = self._ocr(vis_inputs).hidden_states
            image_features = hidden_states[-1]  # [1, 290, 1024] -> [B, seq_len, H]
            image_features = self.image_feature_picker(
                image_features
            )  # [B, seq_len, H] -> [B, T, H]

        # todo: check if this is correct
        n_image_tokens = (vis_inputs["input_ids"] == self.config.image_token_id).sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
            inputs_embeds.device
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        # todo: cache image features
        return inputs_embeds

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        vis_inputs: Optional[Dict] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        run_type: Optional[str] = "ocr",
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        assert run_type in ["qa", "ocr"]

        if run_type == "ocr":
            return self._ocr(vis_inputs)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if hasattr(self, "vis"):
            inputs_embeds = self._merge_image_features(
                inputs_embeds, input_ids, vis_inputs
            )

        outputs = self.llm(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        run_type = kwargs.get("run_type", "ocr")
        if run_type == "ocr":
            vis_inputs = kwargs.pop("vis_inputs")
            kwargs.pop("run_type")
            return self.vis.generate(**vis_inputs, **kwargs)
        else:
            return super().generate(**kwargs)

    @classmethod
    def from_pretrained(
        cls: Type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        llm_model_name: Optional[Union[str, os.PathLike]] = None,
        vis_model_name: Optional[Union[str, os.PathLike]] = None,
        model_part: Optional[str] = None,
        ocr_config: Optional[dict] = None,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        assert (
            pretrained_model_name_or_path is not None
            or llm_model_name is not None
            or vis_model_name is not None
        ), (
            "You have to specify either a pretrained model name or a path to a pretrained model"
        )

        config = (
            TabQConfig.from_pretrained(pretrained_model_name_or_path)
            if pretrained_model_name_or_path
            else TabQConfig()
        )

        if model_part is not None:
            config.model_part = model_part

        if pretrained_model_name_or_path is not None:
            model = cls(config, ocr_config)
            model.load_weight_from_safetensors(pretrained_model_name_or_path)
            # model = super().from_pretrained(
            #     pretrained_model_name_or_path,
            #     config=config,
            #     ocr_config=ocr_config,
            #     **kwargs,
            # )
            # # vis_model_name = os.path.join(pretrained_model_name_or_path, "vis")
            # # model.load_weight(vis_model_name=vis_model_name)
        else:
            config.model_part = ""  # disable init via config
            model = cls(config)
            model.load_weight(llm_model_name, vis_model_name, ocr_config)
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        **kwargs,
    ):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.config.save_pretrained(save_directory, is_main_process=is_main_process)
        self.generation_config.save_pretrained(
            save_directory, is_main_process=is_main_process
        )

        if hasattr(self, "llm"):
            super().save_pretrained(
                os.path.join(save_directory), is_main_process=is_main_process
            )
        if hasattr(self, "vis"):
            self.vis.save_pretrained(
                os.path.join(save_directory, "vis"), is_main_process=is_main_process
            )
