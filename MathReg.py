import json
from utils import read_json, load_prompt_template
from main import args
from pathlib import Path
from API_call import call, call_no_interrupt
from tqdm import tqdm
import statistics
import re

NUM_SAMPLE = float("inf")
def calibrate(output_text: str):
    """
    use regex to extract the mathematic equation and use python to correct answer 
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


    
def generate(model: str, dataset: Path):
    """
    generate answer using MathReg approach
    input: quesitons with decomposed sub-questions
    output: generated subquestions with answers for each question 
    """
    # read data
    data = read_json(dataset)
    generated_data = []
    # load prompt template
    prompt_template = load_prompt_template('./prompts/mathreg_prompt_template.txt')
    max_tokens = 256
    temperature = 0
    # generate
    for i in tqdm(range(min(NUM_SAMPLE, len(data)))):
        question = data[i]
        prompt = prompt_template.format(question=question['question'])
        stop = ['</s>', '\n\n', '\n']
        for i, sub in enumerate(question['sub_questions']):
            prompt += f"\nQuestion 5.{i+1}: " + sub + f"\nAnswer 5.{i+1}: "
            output_text = call_no_interrupt(prompt, model, max_tokens, temperature, args.top_k, args.top_p, args.repetition_penalty, stop)
            # use regex + python to correct the mathematic calculation 
            # corrected_output_text = calibrate(output_text)
            corrected_output_text = output_text
            
            prompt += corrected_output_text
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
    generate_subquestions_answers = generate(model, dataset)
    questions = read_json(dataset)[:len(generate_subquestions_answers)]
    assert len(generate_subquestions_answers) == len(questions)

    for q, qa in zip(questions, generate_subquestions_answers):
        q['sub_questions_answers'] = qa

    for q in questions:
        q['answer'] = extract(q['sub_questions_answers'])
    
    model_name = model.split('/')[-1]
    dataset_name = dataset.stem
    ans_name = f"./out/result_{dataset_name}_{model_name}_direct_t0.json"

    with open(ans_name, 'w') as f:
        json.dump(questions, f, indent=4)


if __name__ == "__main__":
    root = Path("./out")
    models = ["togethercomputer/llama-2-7b-chat", "togethercomputer/llama-2-13b-chat"]
    datasets = [root/'llama-2-7b-chat_multiarith_decomp.json', root/'llama-2-7b-chat_gsm8k_decomp.json', 
                root/'llama-2-13b-chat_multiarith_decomp.json', root/'llama-2-13b-chat_gsm8k_decomp.json']
    for model in models:
        for dataset in datasets:
            one_off(model = model, 
                    dataset= dataset)
