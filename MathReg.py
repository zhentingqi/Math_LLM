import json
from utils import read_json, load_prompt_template
from main import args
from pathlib import Path
from API_call import call, call_no_interrupt
from tqdm import tqdm
import statistics
import re


def calibrate(output_text: str):
    """
    use regex to extract the mathematic equation and use python to correct answer 
    """
    equation_regex = r"([A-Za-z\d\.\%\/\*\+\-\$\s]+) = ([\d\.\$\s]+)(?=[A-Za-z]|\b)"
    answer_regex = r"(The answer is [\$\$\$]?)([\d\.\s]+)"

    def evaluate_expression(expression):
        cleaned_expression = expression.replace(' x ', ' * ')
        cleaned_expression = re.sub(r"[A-Za-z]+", '', cleaned_expression).replace('$', '').replace(' ', '').replace('%', '/100')
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
        return f"{expression.strip()} = {unit}{correct_answer}" if correct_answer is not None else match.group(0)

    def update_final_answer(match, correct_answer):
        prefix, existing_answer = match.groups()
        if existing_answer.strip() != str(correct_answer).strip():
            return f"{prefix}{correct_answer}"
        return match.group(0)

    calibrated_text = re.sub(equation_regex, handle_units, output_text)

    match = re.search(equation_regex, output_text)
    if match:
        expression, _ = match.groups()
        correct_answer = evaluate_expression(expression)
        if correct_answer is not None:
            calibrated_text = re.sub(answer_regex, lambda m: update_final_answer(m, correct_answer), calibrated_text)

    calibrated_text = calibrated_text.strip()
    calibrated_text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", calibrated_text)

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
    max_tokens = 128
    temperature = 0.2
    # generate
    for question in tqdm(data):
        prompt = prompt_template.format(question=question['question'])
        stop = ['</s>', '\n\n', '\n']
        for i, sub in enumerate(question['sub_questions']):
            prompt += f"\nQuestion 5.{i+1}: " + sub + f"\nAnswer 5.{i+1}: "
            output_text = call_no_interrupt(prompt, model, max_tokens, temperature, args.top_k, args.top_p, args.repetition_penalty, stop)
            # use regex + python to correct the mathematic calculation 
            corrected_output_text = calibrate(output_text)
            
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
    match = re.search(r"The answer is\s*\$?(\d+(\.\d+)?)", sub_questions_answers)
    
    if match:
        return float(match.group(1))
    else:
        return None
    
    
def one_off(model: str, dataset: Path):
    generate_subquestions_answers = generate(model, dataset)
    questions = read_json(dataset)
    assert len(generate_subquestions_answers) == len(questions)

    for q, qa in zip(questions, generate_subquestions_answers):
        q['sub_questions_answers'] = qa

    for q in questions:
        q['answer'] = extract(q['sub_questions_answers'])
    
    ans_name = str(dataset).replace('.json', '_with_mathreg.json')
    with open(ans_name, 'w') as f:
        json.dump(questions, f, indent=4)



if __name__ == "__main__":
    root = Path("./out")
    one_off(model = "togethercomputer/llama-2-7b-chat", 
            dataset=root/'llama-2-7b-chat_multiarith_decomp.json')
