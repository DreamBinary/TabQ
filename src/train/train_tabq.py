# -*- coding:utf-8 -*-
# @FileName : train.py
# @Time : 2025/1/11 14:55
# @Author : fiv

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import argparse
import importlib
from train_qa_image import train_qa_image
from train_ocr import train_ocr


def train(
    model_args,
    data_args,
    training_args,
):
    training_type = training_args.training_type
    match training_type:
        case "ocr":
            train_ocr(model_args, data_args, training_args)
        case "qa_image":
            train_qa_image(model_args, data_args, training_args)
        case _:
            raise ValueError(f"training type {training_type} is not supported")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--cfg", type=str, required=True)
    args = parse.parse_args()
    sys.path.insert(0, os.path.dirname(args.cfg))
    cfg = importlib.import_module(os.path.basename(args.cfg).replace(".py", ""))
    train(cfg.ModelArguments(), cfg.DataArguments(), cfg.TrainingArguments())
