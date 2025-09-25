# -*- coding:utf-8 -*-
# @FileName : config.py
# @Time : 2025/1/11 15:21
# @Author : fiv

from dataclasses import dataclass, field
from typing import Optional, List

import transformers
import os
import time

OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"output/run-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    vis_model_name_or_path: Optional[str] = field(default="stepfun-ai/GOT-OCR-2.0-hf")
    llm_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B")


@dataclass
class DataArguments:
    data_pkl_path: Optional[str] = field(
        default="./data/train_all_llm.pkl"
    )  # cache for data
    eval_data_pkl_path: Optional[str] = field(default="./data/val_all_llm.pkl")
    dataset_infos: List[dict] = field(
        default_factory=lambda: [
            {
                "type": "ft_llm",
                "name": "B",
                "split": "train",
            },  # consistent with dataset_infos in ./constants.py
        ]
    )
    eval_dataset_infos: Optional[List[dict]] = field(
        default_factory=lambda: [
            {"type": "ft_llm", "name": "B", "split": "val"},
        ]
    )
    test_num: Optional[int] = field(default=1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_type: Optional[str] = field(default="qa_image")
    output_dir: Optional[str] = field(default=OUTPUT_DIR)
    cache_dir: Optional[str] = field(default="./cache")
    err_log_path: Optional[str] = field(default="./err.log")
    max_length: Optional[int] = field(default=2304)

    do_train: Optional[bool] = field(default=True)
    dataloader_num_workers: Optional[int] = field(default=16)
    per_device_train_batch_size: Optional[int] = field(default=2)
    # auto_find_batch_size: Optional[bool] = field(default=True)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=2e-5)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    bf16: Optional[bool] = field(default=True)
    label_names: Optional[List[str]] = field(default_factory=lambda: ["labels"])
    deepspeed: Optional[str] = field(default="src/config/zero2.json")
    # torch_empty_cache_steps: Optional[int] = field(default=200)
    # gradient_checkpointing: Optional[bool] = field(default=True)
    resume_from_checkpoint: Optional[str] = field(default=False)
    num_train_epochs: Optional[int] = field(default=5)
    logging_steps: Optional[int] = field(default=100)

    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=5000)
    per_device_eval_batch_size: Optional[int] = field(default=8)
    eval_on_start: Optional[bool] = field(default=False)
    include_for_metrics: Optional[List[str]] = field(default_factory=lambda: ["loss"])
    batch_eval_metrics: Optional[bool] = field(default=True)
    prediction_loss_only: Optional[bool] = field(default=False)
    metric_for_best_model: Optional[str] = field(default="loss")

    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=5000)
    jit_mode_eval: Optional[bool] = field(default=False)

    report_to: Optional[List[str]] = field(default_factory=lambda: ["tensorboard"])
    seed: Optional[int] = field(default=42)
