# -*- coding:utf-8 -*-
# @FileName : constants.py
# @Time : 2025/1/27 13:27
# @Author : fiv


DATASET_INFO = {  # type -> name -> split
    "ft_vis": {
        "A": {
            "all": {
                "dataset_path": "data/A/A.jsonl",
                "image_dir": "data/A/images",
            },
            "train": {
                "dataset_path": "data/A/A_train.jsonl",
                "image_dir": "data/A/images/train",
            },
            "val": {
                "dataset_path": "data/A/A_val.jsonl",
                "image_dir": "data/A/images/eval",
            },
        },
    },
    "ft_llm": {
        "B": {
            "all": {
                "dataset_path": "data/B/B.jsonl",
                "image_dir": "data/B/images",
            },
            "train": {
                "dataset_path": "data/B/B_train.jsonl",
                "image_dir": "data/B/images/train",
            },
            "val": {
                "dataset_path": "data/B/B_val.jsonl",
                "image_dir": "data/B/images/eval",
            },
        },
    },
}
