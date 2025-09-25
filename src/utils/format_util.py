# -*- coding:utf-8 -*-
# @FileName : format_util.py
# @Time : 2025/04/14 10:48
# @Author : fiv


class TabQFormatter:
    def __init__(self, processor):
        self.processor = processor

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
        return msg

    def raw_msg(self, item):
        conversation = item["conversation"]
        msg = []
        msg.append({"role": "system", "content": self.processor.system_prompt})
        first = True  # only first user has <image> tag
        for c in conversation:
            if first:
                msg.append({"role": "user", "content": c[0]})
                first = False
            else:
                msg.append({"role": "user", "content": self.remove_tag(c[0])})
            msg.append({"role": "assistant", "content": self.remove_tag(c[1])})

        if "<image>" not in msg[1]["content"]:
            msg[1]["content"] = self.template(item, msg[1]["content"])
        else:
            msg[1]["content"] = self.processor.replace_placeholder(
                msg[1]["content"], placeholder="<image>"
            )
        return msg

    def __call__(self, item, split_response=False):
        if split_response:
            msg = self.format_msg(item)
            return msg[:-1], msg[-1]
        else:
            return self.format_msg(item)
