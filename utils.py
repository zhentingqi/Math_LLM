import json


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template
