# -*- coding:utf-8 -*-
# @FileName : train.py
# @Time : 2025/1/11 14:55
# @Author : fiv

import os
import torch
from transformers import (
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from tqdm import tqdm
from src.model import TabQModel, TabQProcessor
from src.dataset import QueryImageDataset
from src.utils import init_logging

logger = init_logging()


class TestCallback(TrainerCallback):
    def __init__(self, test_data: list, processor: TabQProcessor, output_dir: str):
        self.test_data = test_data
        self.processor = processor
        self.output_dir = os.path.join(output_dir, "test_output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_img_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.output_img_dir, exist_ok=True)
        # cp img
        for d in test_data:
            os.system(f"cp {d['filepath']} {self.output_img_dir}")

    def template(self, item, content):
        if item["is_CoT"]:
            return f"""Answer based on the following LaTex table and question:\nLaTex Table:\n```{self.processor.img_full_placeholder}```\nQuestion:\n{content}"""
        else:
            return f"""Answer directly based on the following LaTex table and question without explanation:\nLaTex Table:\n```{self.processor.img_full_placeholder}```\nQuestion:\n{content}"""

    def remove_tag(self, s):
        return s.replace("<image>\n", "").replace("<image>", "")

    def format_msg(self, item):
        conversation = item["conversation"]
        msg = []
        msg.append({"role": "system", "content": self.processor.system_prompt})
        for c in conversation:
            msg.append({"role": "user", "content": self.remove_tag(c[0])})
            msg.append({"role": "assistant", "content": self.remove_tag(c[1])})

        msg[1]["content"] = self.template(item, msg[1]["content"])

        return msg[:-1], msg[-1]

    def on_save(self, args, state, control, **kwargs):
        logger.info("Testing in on_save...")
        model = kwargs.get("model", None)
        if model is None:
            return

        training_mode = model.training
        model.eval()
        device = model.device
        origin_text_list = []
        ans_text_list = []
        generated_text_list = []
        try:
            for d in tqdm(self.test_data, desc="Testing"):
                vis_inputs = self.processor(
                    text=self.processor.generate_text, image=d["filepath"]
                )
                msg, ans = self.format_msg(d)
                text = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(text=text).to(device)

                generated_ids = model.generate(
                    **inputs,
                    vis_inputs=vis_inputs.to(device),
                    run_type="qa",
                    do_sample=False,
                    tokenizer=self.processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=2048,
                    bos_token_id=151643,
                )
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                origin_text_list.append(text)
                ans_text_list.append(ans)
                generated_text_list.append(generated_text)

            steps = state.global_step
            with open(os.path.join(self.output_dir, f"{steps}.txt"), "w") as f:
                for d, o, g, a in zip(
                    self.test_data, origin_text_list, generated_text_list, ans_text_list
                ):
                    f.write(f"Filepath: {d['filepath']}\n")
                    f.write(f"Origin: {o}\n")
                    f.write(f"Generated: {g}\n")
                    f.write(f"Answer: {a}\n\n")

        except Exception as e:
            logger.info(f"Generation failed: {str(e)}")
        finally:
            model.train(training_mode)
        return control


class VisionLanguageCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer, mlm)

    def __call__(self, features, return_tensors="pt"):
        run_type = features[0].get("run_type", "qa")
        vis_inputs = [f["vis_inputs"] for f in features]
        vis_inputs = super().__call__(vis_inputs, return_tensors)
        other_inputs = []
        for f in features:
            for k, v in f.items():
                if k != "vis_inputs" and k != "run_type":
                    other_inputs.append(v)
        other_inputs = super().__call__(other_inputs, return_tensors)
        result = {"vis_inputs": vis_inputs, "run_type": run_type, **other_inputs}
        return result


def train_qa_image(
    model_args,
    data_args,
    training_args,
):
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    if model_args.model_name_or_path is None:
        if model_args.tokenizer_name_or_path is None:
            model_args.tokenizer_name_or_path = model_args.llm_model_name_or_path
        processor = TabQProcessor.generate_pretrained(
            tokenizer_path=model_args.tokenizer_name_or_path,
            processor_path=model_args.vis_model_name_or_path,
            max_length=training_args.max_length,
        )
        model = TabQModel.from_pretrained(
            llm_model_name=model_args.llm_model_name_or_path,
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
            ocr_config={"num_image_tokens": processor.num_image_tokens},
        )
    model.vis.resize_token_embeddings(len(processor.tokenizer))

    train_dataset = QueryImageDataset(
        dataset_infos=data_args.dataset_infos,
        processor=processor,
        data_pkl_path=data_args.data_pkl_path,
        max_length=training_args.max_length,
    )
    eval_dataset = QueryImageDataset(
        dataset_infos=data_args.eval_dataset_infos,
        processor=processor,
        data_pkl_path=data_args.eval_data_pkl_path,
        max_length=training_args.max_length,
    )
    # test_dataset = random.sample(eval_dataset.data, data_args.test_num)

    def _freeze_param(module):
        for param in module.parameters():
            param.requires_grad = False

    _freeze_param(model.vis)
    model.vis.eval()

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0,
        )
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VisionLanguageCollator(processor.tokenizer),
        callbacks=callbacks,
        # callbacks=[TestCallback(test_dataset, processor, training_args.output_dir)],
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
