# -*- coding:utf-8 -*-
# @FileName : img_ltx_pair_dataset.py
# @Time : 2025/03/06 21:44
# @Author : fiv

from .base_dataset import BaseDataset
import random


class ImgLtxPairDataset(BaseDataset):
    def __init__(
        self, processor, dataset_infos=None, data=None, data_pkl_path=None, **kwargs
    ):
        super(ImgLtxPairDataset, self).__init__(dataset_infos, data, data_pkl_path)
        self.processor = processor
        self.max_length = kwargs.get("max_length", 2048)

    def __getitem__(self, idx):
        flag = True
        while flag:  # void error
            try:
                item = super(ImgLtxPairDataset, self).__getitem__(idx)
                text = self.processor.apply_chat_template(
                    [
                        {"role": "system", "content": self.processor.system_prompt},
                        {
                            "role": "user",
                            "content": f"{self.processor.img_full_placeholder}OCR with LaTex format:",
                        },
                        {"role": "assistant", "content": item["latex"]},
                    ],
                    tokenize=False,
                )
                result = self.processor(
                    text=text, image=item["filepath"], max_length=self.max_length
                )
                for k, v in result.items():
                    result[k] = v.squeeze(0)
                flag = False
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                idx = random.randint(0, len(self.data) - 1)
        return {"vis_inputs": result, "run_type": "ocr"}


if __name__ == "__main__":
    from src.config.arguments_ocr import DataArguments
    from src.model.tabq.processing_tabq import TabQProcessor

    d = DataArguments()
    processor = TabQProcessor.from_pretrained("./tmp")
    imgLtxPairDataset = ImgLtxPairDataset(d.dataset_infos, processor)
    t = imgLtxPairDataset[0]
    print(t)
