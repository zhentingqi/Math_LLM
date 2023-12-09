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
    pass

    
def generate(model: str, dataset: Path):
    """
    generate answer using MathReg approach
    input: quesitons with decomposed sub-questions
    output: generated subquestions with answers for each question 
    """
    # read data
    filename = f"./out/{model}_{dataset}_decomp.json"
    data = read_json(filename)
    generated_data = []
    answers = []
    # load prompt template
    prompt_template = load_prompt_template('chat_prompt_template.txt')
    max_tokens = 128
    temperature = 0.2
    # generate
    for question in tqdm(data):
        prompt = prompt_template.format(question=question['question'])
        stop = ['</s>', '\n\n']
        for i, sub in enumerate(question['sub_questions']):
            prompt += f"\n    # Q.{i+1}: " + sub
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
    answer_with_symbol = sub_questions_answers.split("The answer is: ")[-1]
    pattern = r"^([0-9]*\.?[0-9]+)"
    match = re.match(pattern, answer_with_symbol)
    return float(match.group(1)) if match else None
    
    


    
def one_off(model: str, dataset: Path):
    answers = []
    generate_subquestions_answers = generate(model, dataset)
    questions = read_json(dataset)
    assert len(generate_subquestions_answers) == len(questions)

    questions_with_sub_qs_and_as = []
    for q, qa in zip(questions, generate_subquestions_answers):
        q['sub_questions_answers'] = qa
        questions_with_sub_qs_and_as.append(q)

    # save mathreg output to file
    with open(f"mathreg_out_{model}_{dataset}.json", "w") as f:
        json.dump(questions_with_sub_qs_and_as, f)

    for question in questions_with_sub_qs_and_as:
        answers.append(
            extract(question['sub_questions_answers'])
        )
    return answers


if __name__ == "__main__":
    pass 
