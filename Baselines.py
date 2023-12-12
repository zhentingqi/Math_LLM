import json
from utils import read_json, load_prompt_template
from main import args
from pathlib import Path
from API_call import call, call_no_interrupt
from tqdm import tqdm
import statistics
import re

NUM_SAMPLE = float("inf")
# NUM_SAMPLE = 3

def generate(model: str, dataset: Path, zeroshot: bool):
    """
    input: quesitons with original sub-questions
    output: generated answers for each question 
    """
    # read data
    data = read_json(dataset)
    generated_data = []
    # load prompt template
    if zeroshot:
        prompt_template = load_prompt_template('./prompts/zeroshot_prompt_template.txt')
    else:
        prompt_template = load_prompt_template('./prompts/4-shot_prompt_template.txt')
    max_tokens = 512
    temperature = 0.2
    # generate
    for i in tqdm(range(min(NUM_SAMPLE, len(data)))):
        question = data[i]
        prompt = prompt_template.format(question=question['question'])
        stop = ['</s>', '\n\n']
        output_text = call_no_interrupt(prompt, model, max_tokens, temperature, args.top_k, args.top_p, args.repetition_penalty, stop)
        prompt += output_text
        # save to file
        generated_data.append(prompt)

    return generated_data


def extract(sub_questions_answers: str, zeroshot: bool):
    """
    extract answer 
    input: quesitons with decomposed sub-questions
    output: generated subquestions with answers for each question 
    """
    if not zeroshot:
        matches = re.findall(r"The answer is\s*\$?(\d+(\.\d+)?)", sub_questions_answers)
        if matches:
            return float(matches[-1][0])
        else:
            return None
    else:
        numbers = re.findall(r'\d+(?:\.\d+)?', sub_questions_answers)
        return float(numbers[-1]) if numbers else None
    
    
def one_off(model: str, dataset: Path, zeroshot: bool, dataset_name):
    generate_answers = generate(model, dataset, zeroshot)
    questions = read_json(dataset)[:len(generate_answers)]
    assert len(generate_answers) == len(questions)

    for q, qa in zip(questions, generate_answers):
        q['model_response'] = qa

    for q in questions:
        q['model_answer'] = extract(q['model_response'], zeroshot)
    
    model_name = model.split('/')[-1]
    if zeroshot:
        ans_name = f"./out/result_{dataset_name}_{model_name}_zeroshot.json"
    else:
        ans_name = f"./out/result_{dataset_name}_{model_name}_4-shot.json"

    with open(ans_name, 'w') as f:
        json.dump(questions, f, indent=4)


if __name__ == "__main__":
    zeroshot = False
    root = Path("./data")
    # models = ["togethercomputer/llama-2-7b-chat", "togethercomputer/llama-2-13b-chat", "togethercomputer/llama-2-70b-chat"]
    models = ["togethercomputer/llama-2-7b-chat", "togethercomputer/llama-2-13b-chat"]
    dataset_name = "gsm8k"
    datasets = [root/'gsm8k/test_with_ids.json']
    for model in models:
        for dataset in datasets:
            one_off(model = model, 
                    dataset= dataset,
                    zeroshot = zeroshot,
                    dataset_name=dataset_name)
