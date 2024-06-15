#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/5/31 13:56
# @Author   : 剑客阿良_ALiang
# @File     : ChatTTS_UI.py
# @Blog     : https://huyi-aliang.blog.csdn.net/
# import torch

from ChatTTS import ChatTTS
from IPython.display import Audio
import gradio as gr
import uuid

chat = ChatTTS.Chat()
chat.load_models(source='local', local_path='./pzc163/chatTTS')


def hy(iuput):
    # inputs_cn = """
    # 这是正常的说话是这样子[uv_break]。
    # 带笑的说话是这样[laugh]，可以听出区别吗。
    # """.replace('\n', '')
    inputs_cn = iuput.replace('\n', '')

    params_refine_text = {
        'prompt': '[uv_break][laugh_0][break_4]'
    }
    audio_array_cn = chat.infer(inputs_cn, params_refine_text=params_refine_text)
    # audio_array_en = chat.infer(inputs_en, params_refine_text=params_refine_text)
    audio = Audio(audio_array_cn[0], rate=24_000, autoplay=True)
    audio_data = audio.data

    with open('{}.wav'.format(uuid.uuid4()), 'wb') as f:  # 更改扩展名以匹配原始音频格式
        f.write(audio_data)
    return f.name


if __name__ == '__main__':
    demo = gr.Interface(title='阿良ChatTTS', fn=hy,
                        inputs=[gr.Textbox(lines=4,
                                           placeholder="这是正常的说话是这样子[uv_break]。带笑的说话是这样[laugh]，可以听出区别吗。",
                                           label="文本内容（可以使用[laugh],[uv_break],[lbreak]三类标签）"), ],

                        outputs=gr.Audio(value='音频', type="filepath"))
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
