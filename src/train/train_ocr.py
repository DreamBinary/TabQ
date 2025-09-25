# -*- coding:utf-8 -*-
# @FileName : train_ocr.py
# @Time : 2025/03/15 15:38
# @Author : fiv

import os
import random
from transformers import (
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    EvalPrediction,
)
from transformers.utils import logging
import torch
from tqdm import tqdm
from src.model import TabQModel, TabQProcessor
from src.dataset import ImgLtxPairDataset
from src.utils import init_logging, calculate_all_metrics

logger = init_logging()


class TestCallback(TrainerCallback):
    def __init__(
        self, test_data: list, processor: TabQProcessor, output_dir: str, **kwargs
    ):
        self.test_data = test_data
        self.processor = processor
        self.output_dir = os.path.join(output_dir, "test_output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_img_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.output_img_dir, exist_ok=True)
        # cp img
        for d in test_data:
            os.system(f"cp {d['filepath']} {self.output_img_dir}")
        self.max_length = kwargs.get("max_length", 2048)

    def on_save(self, args, state, control, **kwargs):
        logger.info("Testing in on_save...")
        model = kwargs.get("model", None)
        if model is None:
            return

        training_mode = model.training
        model.eval()
        device = model.device
        origin_text_list = []
        generated_text_list = []
        answer_list = []
        all_metric = []
        try:
            for d in tqdm(self.test_data, desc="Testing"):
                inputs = self.processor(
                    text=self.processor.generate_text, image=d["filepath"]
                )

                generated_ids = model.generate(
                    vis_inputs=inputs.to(device),
                    run_type="ocr",
                    do_sample=False,
                    tokenizer=self.processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=self.max_length,
                )
                generated_text = self.processor.batch_decode(
                    generated_ids[0, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                if isinstance(generated_text, list):
                    generated_text = "".join(generated_text)
                origin_text_list.append(self.processor.generate_text)
                generated_text_list.append(generated_text)
                answer_list.append(d["latex"])
                metric = calculate_all_metrics(generated_text, d["latex"])
                all_metric.append(metric)

            steps = state.global_step
            with open(os.path.join(self.output_dir, f"{steps}.txt"), "w") as f:
                for d, o, g, m in zip(
                    self.test_data, origin_text_list, generated_text_list, all_metric
                ):
                    f.write(f"Filepath: {d['filepath']}\n")
                    f.write(f"Origin: {o}\n")
                    f.write(f"Generated: {g}\n")
                    f.write(f"Answer: {d['latex']}\n")
                    f.write(f"Metric: {m}\n\n")

        except Exception as e:
            logger.info(f"Generation failed: {str(e)}")
        finally:
            model.train(training_mode)
        return control


class VisionCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer, mlm)

    def __call__(self, features, return_tensors="pt"):
        vis_inputs = [f["vis_inputs"] for f in features]
        vis_inputs = super().__call__(vis_inputs, return_tensors)
        result = {"vis_inputs": vis_inputs}
        first = features[0]
        for k, v in first.items():
            if k != "vis_inputs":
                result[k] = v
        for k, v in vis_inputs.items():
            result[k] = v
        # label_name = "labels"
        # result[label_name] = vis_inputs[label_name]
        return result


all_metric = []


def compute_metrics(result: EvalPrediction, compute_result=False, processor=None):
    logits = result.predictions[0]
    pred_id = logits.argmax(axis=-1)
    label_id = result.label_ids

    mask = label_id != -100
    pred_id = pred_id[mask]
    label_id = label_id[mask]

    pred_str = processor.batch_decode(pred_id, skip_special_tokens=True)
    label_str = processor.batch_decode(label_id, skip_special_tokens=True)
    print(f"pred_str: {pred_str}")
    print(f"label_str: {label_str}")
    metric = {}
    metric.update(calculate_all_metrics(pred_str, label_str))
    if result.losses is not None:
        metric["loss"] = sum(result.losses) / len(result.losses)
    logging.info(f"metric: {metric}")
    all_metric.append(metric)
    if compute_result:
        # average all metrics
        avg_metric = {}
        for k in metric.keys():
            avg_metric[k] = sum([m[k] for m in all_metric]) / len(all_metric)
        all_metric.clear()
        logging.info(f"avg_metric: {avg_metric}")
        return avg_metric
    else:
        return metric


def train_ocr(
    model_args,
    data_args,
    training_args,
):
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    if model_args.model_name_or_path is None:
        processor = TabQProcessor.generate_pretrained(
            tokenizer_path=model_args.tokenizer_name_or_path,
            processor_path=model_args.vis_model_name_or_path,
            max_length=training_args.max_length,
        )
        model = TabQModel.from_pretrained(
            vis_model_name=model_args.vis_model_name_or_path,
            ocr_config={"num_image_tokens": processor.num_image_tokens},
        )
        model.config.image_token_id = processor.img_token_id
    else:
        processor = TabQProcessor.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
        )
        model = TabQModel.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
        )
    model.vis.resize_token_embeddings(len(processor.tokenizer))

    train_dataset = ImgLtxPairDataset(
        dataset_infos=data_args.dataset_infos,
        processor=processor,
        data_pkl_path=data_args.data_pkl_path,
        max_length=training_args.max_length,
    )
    eval_dataset = ImgLtxPairDataset(
        dataset_infos=data_args.eval_dataset_infos,
        processor=processor,
        data_pkl_path=data_args.eval_data_pkl_path,
        max_length=training_args.max_length,
    )
    test_dataset = random.sample(eval_dataset.data, data_args.test_num)

    # def _freeze_param(module):
    #     for param in module.parameters():
    #         param.requires_grad = False

    # _freeze_param(model.vision_encoder)
    def compute_metrics_fn(result: EvalPrediction, compute_result=False):
        return compute_metrics(
            result,
            compute_result=compute_result,
            processor=processor,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VisionCollator(processor.tokenizer),
        callbacks=[TestCallback(test_dataset, processor, training_args.output_dir)],
        # compute_metrics=compute_metrics_fn,
    )

    try:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    except Exception as e:
        torch.cuda.empty_cache()
        trainer._save_checkpoint(
            model=trainer.model,
            trial=None,
        )
        logger.info(f"Training failed: {str(e)}")
        # write to training_args.err_log_path
        with open(training_args.err_log_path, "a") as f:
            f.write(f"{str(e)}\n")
        raise e

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"model saved to {training_args.output_dir}")
