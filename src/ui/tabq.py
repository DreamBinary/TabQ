# -*- coding:utf-8 -*-
# @FileName : tabq.py
# @Time : 2025/04/15 17:54
# @Author : fiv

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from PIL import Image
import gradio as gr
from src.infer import (
    init_tabq,
    init_tabq_vis,
    infer_tabq_vis_stream,
    infer_tabq_stream,
)
from src.utils import TabQFormatter

model, processor = None, None


def ocr_fun(img_arr):
    img = Image.fromarray(img_arr, mode="RGB")
    result = ""
    it = infer_tabq_vis_stream(model, processor, img)
    for r in it:
        result += r
        yield result


def preprocess(question):
    item = {"conversation": [[question, "ans"]], "is_CoT": True}
    text, ans = formatter(item, split_response=True)
    text = formatter.processor.apply_chat_template(
        text, tokenize=False, add_generation_prompt=True
    )
    return text


def llm_fun(img_arr, question, kwargs):
    """
    kwargs: {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.2}
    """
    try:
        kwargs = kwargs = json.loads(kwargs)
    except Exception:
        kwargs = {}
    img = Image.fromarray(img_arr, mode="RGB")
    result = ""
    it = infer_tabq_stream(
        model, processor, {"image": img, "text": preprocess(question)}, **kwargs
    )
    for r in it:
        result += r
        yield result


def ocr_page():
    with gr.Blocks() as ocr:
        with gr.Column():
            gr.Markdown(
                """
                # TabQ
                ## ocr table to latex
                """
            )
            with gr.Row():
                inp = gr.Image(
                    label="Input Table Image",
                    placeholder="Upload an table image...",
                )
                out = gr.Text(
                    label="Output Text",
                    lines=15,
                    placeholder="Output text will be displayed here...",
                )
        btn = gr.Button("Run")
        btn.click(
            fn=ocr_fun,
            inputs=[inp],
            outputs=[out],
        )
    return ocr


def llm_page():
    with gr.Blocks() as llm:
        with gr.Column():
            gr.Markdown(
                """
                # TabQ
                ## table understanding
                """
            )
            with gr.Row():
                inp_img = gr.Image(
                    label="Input Table Image",
                    placeholder="Upload an table image...",
                )
                with gr.Column():
                    kwargs_text = gr.Textbox(
                        label="Input Setting",
                        placeholder="Enter your text here...",
                    )
                    inp_text = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter your text here...",
                    )
                    out = gr.Text(
                        label="Output Text",
                        placeholder="Output text will be displayed here...",
                    )
        btn = gr.Button("Run")
        btn.click(
            fn=llm_fun,
            inputs=[inp_img, inp_text, kwargs_text],
            outputs=[out],
        )
    return llm


def main(port=9878):
    tab_iface = gr.TabbedInterface(
        [ocr_page(), llm_page()],
        ["TabQ-OCR", "TabQ-Understand"],
    )
    tab_iface.launch(inline=False, inbrowser=True, server_port=port, share=True)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_type", type=str, default="llm", choices=["ocr", "llm"])
    parser.add_argument(
        "--vis_model_name_or_path", type=str, default="stepfun-ai/GOT-OCR-2.0-hf"
    )
    parser.add_argument(
        "--llm_model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="tabq_pretrained_model_name_or_path",
    )
    parser.add_argument("--max_length", type=int, default=2304)
    parser.add_argument("--tokenizer_name_or_path", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument(
        "--port", type=int, default=9878, help="Port for the Gradio app"
    )
    args = parser.parse_args()
    if args.run_type == "llm":
        model, processor = init_tabq(args)
        formatter = TabQFormatter(processor)
    else:
        model, processor = init_tabq_vis(
            args.vis_model_name_or_path, args.tokenizer_name_or_path
        )
    print("Model and processor initialized successfully.")
    main(args.port)
