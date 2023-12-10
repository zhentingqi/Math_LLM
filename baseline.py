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
    
def generate(model: str, dataset: Path):
    """
    input: quesitons with original sub-questions
    output: generated answers for each question 
    """
    # read data
    data = read_json(dataset)
    generated_data = []
    # load prompt template
    prompt_template = load_prompt_template('./prompts/fewshot_prompt_template.txt')
    max_tokens = 128
    temperature = 0.2
    # generate
    for i in tqdm(range(min(NUM_SAMPLE, len(data)))):
        question = data[i]
        prompt = prompt_template.format(question=question['question'])
        stop = ['</s>', '\n\n', '\n']
        output_text = call_no_interrupt(prompt, model, max_tokens, temperature, args.top_k, args.top_p, args.repetition_penalty, stop)
        prompt += output_text
        # save to file
        generated_data.append(prompt)

    return generated_data


def extract(sub_questions_answers: str):
    """
    extract answer 
    input: quesitons with decomposed sub-questions
    output: generated subquestions with answers for each question 
    """
    matches = re.findall(r"The answer is\s*\$?(\d+(\.\d+)?)", sub_questions_answers)
    if matches:
        return float(matches[-1][0])
    else:
        return None
    
    
def one_off(model: str, dataset: Path):
    generate_answers = generate(model, dataset)
    questions = read_json(dataset)[:len(generate_answers)]
    assert len(generate_answers) == len(questions)

    for q, qa in zip(questions, generate_answers):
        q['model_response'] = qa

    for q in questions:
        q['model_answer'] = extract(q['model_response'])
    
    model_name = model.split('/')[-1]
    dataset_name = dataset.stem
    ans_name = f"./out/result_{dataset_name}_{model_name}_fewshot.json"

    with open(ans_name, 'w') as f:
        json.dump(questions, f, indent=4)


if __name__ == "__main__":
    root = Path("./data")
    models = ["togethercomputer/llama-2-7b-chat", "togethercomputer/llama-2-13b-chat"]
    datasets = [root/'multiarith/test.json',root/'gsm8k/test_with_ids.json']
    for model in models:
        for dataset in datasets:
            one_off(model = model, 
                    dataset= dataset)
