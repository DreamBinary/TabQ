# -*- coding:utf-8 -*-
# @FileName : query_dataset.py
# @Time : 2025/03/06 21:48
# @Author : fiv

from .base_dataset import BaseDataset
from ..utils import TabQFormatter
import random


class QueryDataset(BaseDataset):
    def __init__(self, dataset_infos, tokenizer):
        super(QueryDataset, self).__init__(dataset_infos)
        self.tokenizer = tokenizer
        self.system_prompt = 'You are a helpful assistant, helping the user with "LaTex Table" related problems.'

    def template(self, item, content):
        if item["is_CoT"]:
            return f"""Answer based on the following LaTex table and question:\nLaTex Table:\n```{item["latex"]}```\nQuestion:\n{content}"""
        else:
            return f"""Answer directly based on the following LaTex table and question without explanation:\nLaTex Table:\n```{item["latex"]}```\nQuestion:\n{content}"""

    def format_item(self, item):
        conversation = item["conversation"]
        msg = []
        msg.append({"role": "system", "content": self.system_prompt})
        for c in conversation:
            msg.append({"role": "user", "content": c[0]})
            msg.append({"role": "assistant", "content": c[1]})
        msg[1]["content"] = self.template(item, msg[1]["content"])
        input_ids = self.tokenizer.apply_chat_template(msg, tokenize=True)
        return {"input_ids": input_ids}

    def __getitem__(self, idx):
        return self.format_item(self.data[idx])


class QueryImageDataset(BaseDataset):
    def __init__(
        self, processor, dataset_infos=None, data=None, data_pkl_path=None, **kwargs
    ):
        super(QueryImageDataset, self).__init__(dataset_infos, data, data_pkl_path)
        self.processor = processor
        self.max_length = kwargs.get("max_length", 4096)
        self.formatter = TabQFormatter(processor)

    def get_vis_inputs(self, item):
        result = self.processor(
            text=self.processor.generate_text, image=item["filepath"]
        )
        for k, v in result.items():
            result[k] = v.squeeze(0)
        return result

    def __getitem__(self, idx):
        while True:
            try:
                item = super(QueryImageDataset, self).__getitem__(idx)
                vis_inputs = self.get_vis_inputs(item)
                msg = self.formatter(item)
                text = self.processor.apply_chat_template(msg, tokenize=False)
                result = self.processor(text=text, max_length=self.max_length)
                for k, v in result.items():
                    result[k] = v.squeeze(0)
                result = {"vis_inputs": vis_inputs, "run_type": "qa", **result}
                break
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                idx = random.randint(0, len(self.data) - 1)
        return result


if __name__ == "__main__":
    from src.config.arguments_image import DataArguments
    from src.model.tabq.processing_tabq import TabQProcessor

    d = DataArguments()
    processor = TabQProcessor.from_pretrained("./tmp")
    QqueryDataset = QueryImageDataset(processor, d.dataset_infos)
    print(QqueryDataset[0])
