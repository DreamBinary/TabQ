# -*- coding:utf-8 -*-
# @FileName : processing_tabq.py
# @Time : 2025/03/01 19:30
# @Author : fiv


from transformers import AutoTokenizer, GotOcr2Processor
from transformers.image_processing_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.image_utils import load_images
import logging

logger = logging.getLogger(__name__)


class TabQProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "GotOcr2ImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        if chat_template is None and tokenizer is not None:
            chat_template = tokenizer.chat_template
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_pad_token = "<imgpad>"
        self.end_token = "<|im_end|>"
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(self.img_pad_token)
        self.num_image_tokens = 256
        self.img_placeholder = "<image>"
        self.img_full_placeholder = f"{self.img_start_token + self.img_pad_token * self.num_image_tokens + self.img_end_token}\n"

        self.system_prompt = 'You are a helpful assistant, helping the user with "LaTex Table" related problems.'

        self.generate_text = self.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"{self.img_full_placeholder}OCR with LaTex format:",
                },
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

    def __call__(
        self, image=None, text=None, max_length=None, **kwargs
    ) -> BatchFeature:
        if text is not None and not isinstance(text, list):
            text = [text]
        if image is not None and not isinstance(image, list):
            image = [image]

        if text is not None and image is not None and len(image) != len(text):
            raise ValueError(
                f"image and text must have the same length, but got {len(image)} and {len(text)}."
            )
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                # padding=False,
                return_tensors="pt",
                max_length=max_length,
                padding=(max_length is None),
                truncation=(max_length is not None),
            )
        else:
            text_inputs = {}

        if image is not None:
            if isinstance(image[0], str):
                image = load_images(image)
            image_inputs = self.image_processor(images=image, return_tensors="pt")
        else:
            image_inputs = {}
        return BatchFeature(data={**text_inputs, **image_inputs})

    def replace_placeholder(self, text, placeholder=None):
        placeholder = placeholder or self.img_placeholder
        return text.replace(placeholder, self.img_full_placeholder)

    @classmethod
    def generate_pretrained(cls, tokenizer_path, processor_path, max_length=None):
        qwen_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            model_max_length=max_length,
            padding_side="right",
        )
        got_processor = GotOcr2Processor.from_pretrained(
            processor_path, trust_remote_code=True
        )
        got_tokenizer = got_processor.tokenizer
        got_added_vocab = got_tokenizer.get_added_vocab()
        qwen_vocab = qwen_tokenizer.get_vocab()
        for token in got_added_vocab:
            if token not in qwen_vocab:
                qwen_tokenizer.add_tokens(token)
        got_image_processor = got_processor.image_processor
        processor = cls(image_processor=got_image_processor, tokenizer=qwen_tokenizer)
        return processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


if __name__ == "__main__":
    from PIL import Image

    processor = TabQProcessor.generate_pretrained()
    processor.save_pretrained("./tmp")
    image = Image.open("./tmp.png")
    inputs = processor(image, return_tensors="pt").to("cuda")
    outputs = processor.batch_decode(inputs["input_ids"])
    print(inputs)
    print(outputs)
