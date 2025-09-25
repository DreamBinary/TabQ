# -*- coding:utf-8 -*-
# @FileName : base_datasest.py
# @Time : 2025/02/15 11:30
# @Author : fiv

import jsonlines
import pickle
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from src.config.constants import DATASET_INFO
import random

import logging

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, dataset_infos=None, data=None, data_pkl_path=None):
        assert (
            dataset_infos is not None or data is not None or data_pkl_path is not None
        )
        super(BaseDataset, self).__init__()
        if data is not None:
            self.data = data
        elif data_pkl_path is not None and os.path.exists(data_pkl_path):
            self.data = pickle.load(open(data_pkl_path, "rb"))
        else:
            self.dataset_infos = []
            for i in dataset_infos:
                self.dataset_infos.append(
                    DATASET_INFO[i["type"]][i["name"]][i["split"]]
                )

            self.data = self.gen_data()
            if data_pkl_path is not None:
                pickle.dump(self.data, open(data_pkl_path, "wb"))
        # shuffle data
        random.shuffle(self.data)

    def gen_data(self):
        data = []
        for dataset_info in self.dataset_infos:
            latex_map = {}
            if "latex_map_path" in dataset_info:
                with jsonlines.open(dataset_info["latex_map_path"]) as reader:
                    for obj in reader:
                        latex_map.update(obj)
            dataset_path = dataset_info["dataset_path"]
            image_dir = dataset_info["image_dir"]
            with jsonlines.open(dataset_path) as reader:
                reader = list(reader)
                for obj in tqdm(
                    reader, total=len(reader), desc=f"Loading {dataset_path}"
                ):
                    obj["filepath"] = os.path.join(image_dir, obj["filename"])
                    if not os.path.exists(obj["filepath"]):
                        logger.warning(f"File {obj['filepath']} not found, skipping...")
                        continue
                    if (
                        "latex" in obj and obj["latex"] == "" or "latex" not in obj
                    ) and obj["filename"] in latex_map:
                        obj["latex"] = latex_map[obj["filename"]]
                    if isinstance(obj["latex"], list):
                        obj["latex"] = obj["latex"][0]
                    del obj["filename"]
                    data.append(obj)
        logger.info(f"Total data: {len(data)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
