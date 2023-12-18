import json
from utils import read_json, load_prompt_template
from arg_parser import get_parser
from pathlib import Path
from models.TogetherAI_API import call, call_no_interrupt
from tqdm import tqdm
import re


NUM_SAMPLE = float("inf")   #! debug only
# NUM_SAMPLE = 3


def regex_calibrate(output_text: str):
    """
    use regex to extract_answer_from_response the mathematic equation and use python to correct answer 
    """
    equation_regex = r"([\d\.\%\/\*\+\-\$\s]+) = ([\d\.\$\s]+)(?=[A-Za-z,.;!?]|\b)"

    def evaluate_expression(expression):
        cleaned_expression = re.sub(r'\s+\.', '.', expression)
        cleaned_expression = re.sub(r'\.\s+', '.', cleaned_expression)
        cleaned_expression = cleaned_expression.replace(' x ', ' * ').replace('$', '').replace('%', '/100')
        cleaned_expression = re.sub(r'\s+', '', cleaned_expression)
        try:
            return eval(cleaned_expression, {}, {})
        except Exception:
            return None

    def handle_units(match):
        expression, current_answer = match.groups()
        unit = re.findall(r"[\$\$\$]", current_answer)
        unit = unit[0] if unit else ''
        correct_answer = evaluate_expression(expression)
        if correct_answer is None:
            return match.group(0)
        if '.' in current_answer or correct_answer % 1 != 0:
            correct_answer = f"{correct_answer:.6f}"
        else:
            correct_answer = int(correct_answer)
        return f" {expression.strip()} = {unit}{correct_answer} " if correct_answer is not None else match.group(0)

    calibrated_text = re.sub(equation_regex, handle_units, output_text)
    calibrated_text = calibrated_text.strip()

    calibrated_text = re.sub(r"(\d)([A-Za-z,.;!?])", r"\1 \2", calibrated_text)
    calibrated_text = re.sub(r"(\d)\s+(\.\d+)", r"\1\2", calibrated_text)

    return calibrated_text


def generate(args, model: str, dataset: Path):
    """
    input: questions with original sub-questions
    output: generated answers for each question 
    """
    # read data
    data = read_json(dataset)
    generated_data = []
    # load prompt template
    if args.zeroshot:
        prompt_template = load_prompt_template('./prompts/zeroshot_prompt_template.txt')
    else:
        prompt_template = load_prompt_template('./prompts/8-shot_cot_prompt.txt')
    max_tokens = 512
    temperature = 0.2
    # generate
    for i in tqdm(range(min(NUM_SAMPLE, len(data)))):
        question = data[i]
        prompt = prompt_template.format(question=question['question'])
        stop = ['</s>', '\n\n']
        output_text = call_no_interrupt(prompt, model, max_tokens, temperature, args.top_k, args.top_p, args.repetition_penalty, stop)
        # output_text = regex_calibrate(output_text)
        
        prompt += output_text
        # save to file
        generated_data.append(prompt)

    return generated_data


def extract_answer_from_response(response: str, zeroshot: bool):
    """
    extract_answer_from_response answer 
    input: response
    output: generated response with answers for each question 
    """
    # few-shot, find the answer after "The answer is"
    if not zeroshot:
        matches = re.findall(r"The answer is\s*\$?(\d+(\.\d+)?)", response)
        if matches:
            return float(matches[-1][0])
        else:
            return None
    else:
    # zeroshot, find the last number in the response as the answer 
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        return float(numbers[-1]) if numbers else None
    
    
def do_few_shot_cot_generation(args, model: str, dataset_name: str, dataset: Path):
    generate_answers = generate(args, model, dataset)
    questions = read_json(dataset)[:len(generate_answers)]
    assert len(generate_answers) == len(questions)

    for q, qa in zip(questions, generate_answers):
        q['model_response'] = qa

    for q in questions:
        q['model_answer'] = extract_answer_from_response(q['model_response'], args.zeroshot)
    
    model_name = model.split('/')[-1]
    if args.zeroshot:
        ans_name = f"./out/baselines/result_{dataset_name}_{model_name}_zeroshot.json"
    else:
        ans_name = f"./out/baselines/result_{dataset_name}_{model_name}_4-shot.json"

    with open(ans_name, 'w') as f:
        json.dump(questions, f, indent=4)


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--zeroshot", action='store_true')
    args = parser.parse_args()
    
    models = ["togethercomputer/llama-2-7b-chat",]
    dataset_names = ["SVAMP",]
    root = Path("./data")
    datasets = [root/f'{dataset_name}/test_with_ids.json' for dataset_name in dataset_names]
    
    for model in models:
        for dataset_name, dataset in zip(dataset_names, datasets):
            do_few_shot_cot_generation(
                args = args,
                model = model, 
                dataset_name = dataset_name,
                dataset= dataset,
            )
