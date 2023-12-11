import together
from typing import List, Dict
import os
import time

keys = [
    "fb9262df6fe23a25f35355b42cc4d3182754fd9d186309aad0cd4f52fe9c8306",
    'df2fbd8b27413e6a3b5d87a8df44c0ccfc7a175d19445a5f92a9a29450c1ec53',
    '3668e80e292c32d8a23ed01bddc8d4efabc786ede0ea7d40d2de3c8ca49d6015',
    '7b1925f49e33f63f6d64c26a6bbf78ec4373e8d5df205ac785573311a4acd491',
    'd16049df3f5732d9bf180f9b0d79947d6c20471bc9d9277166c5b6e180cef5cf',
    '1d328918b6c9af31dfafc1c6751890e52481c75866448c9db364a7e58c666814',
    '31a5be9a66b0bf3e10d877640b7ae5f73464949c157d7d01cdb6de1f16f53fbb',
    'd202990ef50e94b0df994bc9d35f75cfe90f91f4aa126b08e33dd391d3e4a4e6'
]
together.api_key = keys[0]
model_list = [d['name'] for d in together.Models.list()]

cnt = 0

def call(prompt, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop):
    global cnt
    assert model in model_list, f"model should be one of {model_list}"

    response = together.Complete.create(
        prompt = prompt, 
        model = model, 
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = top_k,
        top_p = top_p,
        repetition_penalty = repetition_penalty,
        api_key=keys[cnt % len(keys)],
        stop = stop
    )
    cnt = (cnt + 1) % len(keys)
    # return text
    return response['output']['choices'][0]['text']


def call_no_interrupt(prompt, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop):
    output = None
    while output is None:
        try:
            output = call(prompt, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop)
        except:
            time.sleep(10)

    return output


def _test():
    output = call(
        prompt = "<human>: What are Isaac Asimov's Three Laws of Robotics?\n<bot>:", 
        model = "togethercomputer/llama-2-7b", 
        max_tokens = 256,
        temperature = 0.8,
        top_k = 60,
        top_p = 0.6,
        repetition_penalty = 1.1,
        stop = ['<human>', '\n\n']
    )
    print(output)


if __name__ == "__main__":
    print(model_list)
    _test()