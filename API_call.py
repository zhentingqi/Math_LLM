import together
from typing import List, Dict
import time


keys = [
    "fb9262df6fe23a25f35355b42cc4d3182754fd9d186309aad0cd4f52fe9c8306",
]       # Add your togetherAI API keys here

together.api_key = keys[0]
model_list = [d['name'] for d in together.Models.list()]

cnt = 0


def call(prompt, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop):
    global cnt
    assert model in model_list, f"model should be one of {model_list}"

    response = together.Complete.create(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        api_key=keys[cnt % len(keys)],
        stop=stop
    )
    cnt = (cnt + 1) % len(keys)
    # return text
    return response['output']['choices'][0]['text']


def call_no_interrupt(prompt, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop):
    output = None
    while output is None:
        try:
            output = call(prompt, model, max_tokens, temperature,
                          top_k, top_p, repetition_penalty, stop)
        except:
            time.sleep(10)

    return output


def _test():
    output = call(
        prompt="<human>: What are Isaac Asimov's Three Laws of Robotics?\n<bot>:",
        model="togethercomputer/llama-2-7b",
        max_tokens=256,
        temperature=0.8,
        top_k=60,
        top_p=0.6,
        repetition_penalty=1.1,
        stop=['<human>', '\n\n']
    )
    print(output)


if __name__ == "__main__":
    print(model_list)
    _test()
