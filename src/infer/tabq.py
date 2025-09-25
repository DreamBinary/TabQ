# -*- coding:utf-8 -*-
# @FileName : tabq.py
# @Time : 2025/04/06 13:59
# @Author : fiv

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import torch
from transformers import TextIteratorStreamer
from src.model import TabQProcessor, TabQModel
from threading import Thread
import random
from time import sleep

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_tabq_vis(
    model_name_or_path="stepfun-ai/GOT-OCR-2.0-hf", tokenizer_path="Qwen/Qwen2-0.5B"
):
    # processor = TabQProcessor.from_pretrained(
    #     pretrained_model_name_or_path=model_name_or_path,
    # )
    processor = TabQProcessor.generate_pretrained(
        tokenizer_path=tokenizer_path,
        processor_path=model_name_or_path,
        max_length=2048,
    )
    model = TabQModel.from_pretrained(
        vis_model_name=model_name_or_path,
        ocr_config={"num_image_tokens": processor.num_image_tokens},
    )
    model.config.image_token_id = processor.img_token_id
    model.to(device)
    return model, processor


def infer_tabq_vis(model, processor, images):
    if not isinstance(images, list):
        images = [images]

    texts = [processor.generate_text for _ in range(len(images))]
    inputs = processor(
        text=texts,
        image=images,
    )

    generate_ids = model.generate(
        vis_inputs=inputs.to(device),
        run_type="ocr",
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=2048,
    )
    input_length = inputs["input_ids"].shape[1]
    batch_size = generate_ids.shape[0]
    results = []
    for i in range(batch_size):
        generated_tokens = generate_ids[i, input_length:]
        decoded_text = processor.decode(generated_tokens, skip_special_tokens=True)
        results.append(decoded_text)
    return results


def infer_tabq_vis_stream(model, processor, images):
    if not isinstance(images, list):
        images = [images]
    texts = [processor.generate_text for _ in range(len(images))]
    inputs = processor(
        text=texts,
        image=images,
    )
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generate_kwargs = {
        "vis_inputs": inputs.to(device),
        "run_type": "ocr",
        "do_sample": False,
        "tokenizer": processor.tokenizer,
        "stop_strings": ["<|im_end|>", r"end{tabular}"],
        "max_new_tokens": 2048,
        "streamer": streamer,
    }
    thread = Thread(
        target=model.generate,
        kwargs=generate_kwargs,
    )
    thread.start()
    for text in streamer:
        yield text


def init_tabq(args):
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = (
            args.vis_model_name_or_path or args.pretrained_model_name_or_path
        )
    processor = TabQProcessor.generate_pretrained(
        tokenizer_path=args.tokenizer_name_or_path,
        processor_path=args.vis_model_name_or_path,
        # max_length=args.max_length,
    )
    model = TabQModel.from_pretrained(
        llm_model_name=args.llm_model_name_or_path,
        vis_model_name=args.vis_model_name_or_path,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        ocr_config={"num_image_tokens": processor.num_image_tokens},
    )
    model.config.image_token_id = processor.img_token_id
    model.to(device)
    model.eval()
    return model, processor


def infer_tabq(model, processor, inputs):
    images = inputs["image"]
    texts = inputs["text"]
    if not isinstance(images, list):
        images = [images]
    if not isinstance(texts, list):
        texts = [texts]
    assert len(images) == len(texts), (
        f"images and texts must be same length, but got {len(images)} and {len(texts)}"
    )

    vis_texts = [processor.generate_text for _ in range(len(images))]
    vis_inputs = processor(
        text=vis_texts,
        image=images,
    )
    inputs = processor(text=texts)

    with torch.no_grad():
        generate_ids = model.generate(
            vis_inputs=vis_inputs.to(device),
            **inputs.to(device),
            run_type="qa",
            do_sample=False,
            tokenizer=processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=1024,
        )
    input_length = inputs["input_ids"].shape[1]
    batch_size = generate_ids.shape[0]
    results = []
    for i in range(batch_size):
        generated_tokens = generate_ids[i, input_length:]
        decoded_text = processor.decode(generated_tokens, skip_special_tokens=True)
        results.append(decoded_text)
    return results


def infer_tabq_stream(model, processor, inputs, **kwargs):
    images = inputs["image"]
    texts = inputs["text"]
    if not isinstance(images, list):
        images = [images]
    if not isinstance(texts, list):
        texts = [texts]
    assert len(images) == len(texts), (
        f"images and texts must be same length, but got {len(images)} and {len(texts)}"
    )

    vis_texts = [processor.generate_text for _ in range(len(images))]
    vis_inputs = processor(
        text=vis_texts,
        image=images,
    )
    inputs = processor(text=texts)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generate_kwargs = {
        "vis_inputs": vis_inputs.to(device),
        **inputs.to(device),
        "run_type": "qa",
        "do_sample": False,
        "tokenizer": processor.tokenizer,
        "stop_strings": "<|im_end|>",
        "max_new_tokens": 1024,
        "streamer": streamer,
        **kwargs,
    }
    thread = Thread(
        target=model.generate,
        kwargs=generate_kwargs,
    )
    thread.start()

    for text in streamer:
        yield text


if __name__ == "__main__":
    import argparse
    import os
    from src.utils import TabQFormatter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vis_model_name_or_path", type=str, default="output/ocr_rendered/vis"
    )
    parser.add_argument(
        "--llm_model_name_or_path", type=str, default="cache/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="output/llm_new/checkpoint-15000",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, default="cache/Qwen/Qwen2-0.5B"
    )
    args = parser.parse_args()
    model, processor = init_tabq(args)
    print("-->> model path", args.pretrained_model_name_or_path)

    formatter = TabQFormatter(processor)

    import jsonlines

    with jsonlines.open("data/MMTab/MMTab_format_train.jsonl") as reader:
        reader = list(reader)
        while True:
            item = random.choice(reader)
            # item = r"""{"conversation": [["I need to know the count of rows and columns in this specific table. Format your final answer as a JSON, using the structure {\"row_number\": \"m\", \"column_number\": \"n\"}.", "This table has 3 rows and 3 columns. Thus, the final answer is {\"row_number\": \"3\", \"column_number\": \"3\"}."]], "filename": "TABMWP_21652.jpg", "latex": "", "dataset": "MMTab/TSD", "is_CoT": True}"""
            # item = eval(item
            # prompt = "  请简单介绍一下Qwen-2B。"  # 提示文本
            # messages = [
            #     {"role": "system", "content": "你是一个智能AI助手"},  # 系统角色消息
            #     {"role": "user", "content": prompt}  # 用户角色消息
            # ]
            texts, ans = formatter(item, split_response=True)
            texts = processor.apply_chat_template(
                texts, tokenize=False, add_generation_prompt=True
            )
            image_dir = "data/MMTab/images/train"
            images = os.path.join(image_dir, item["filename"])

            inputs = {
                "image": images,
                "text": texts,
            }
            results = infer_tabq(model, processor, inputs)
            print("Texts:", texts, flush=True)
            print("Results:", results[0], flush=True)
            print("Ans:", ans["content"], flush=True)
            print("Path:", f"./tmp_test_tabq/{item['filename']}", flush=True)
            print("")
            os.system(f"cp {images} ./tmp_test_tabq")
            sleep(15)
