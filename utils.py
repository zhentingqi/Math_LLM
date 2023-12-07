import json
from argparse import ArgumentParser


def get_args():
    parser=ArgumentParser()
    
    parser.add_argument('--max_tokens', type=int, default=256, help='max_tokens')
    parser.add_argument('--temperature', type=float, default=0.8, help='temperature')
    parser.add_argument('--top_k', type=int, default=60, help='top_k')
    parser.add_argument('--top_p', type=float, default=0.6, help='top_p')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='repetition_penalty')
    parser.add_argument('--stop', type=list, default=['</s>'], help='stop')
    args = parser.parse_args()
    return args

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template