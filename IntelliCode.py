import json
from utils import read_json,load_prompt_template
from main import args
from pathlib import Path
from API_call import call, call_no_interrupt
from tqdm import tqdm
import statistics


def post_process(output_text: str, stop: list):
    '''Post-process output code for each step'''
    processed_text = []
    for line in output_text.split("\n"):
        contains_stop = False
        for s in stop:
            if s in line:
                contains_stop = True
                break
        if not contains_stop:
            processed_text.append(line)
    return "\n".join(processed_text)    


def generate(filename: Path = 'decomposition_result.json'):
    """
    generate code for each decomposed subquestion using IntelliCode approach
    input: quesitons with decomposed sub-questions
    output: generated Python code for all the quesitons
    """
    model = "togethercomputer/CodeLlama-34b-Python"

    # read data
    data = read_json(filename)
    generated_data = []
    # load prompt template
    prompt_template = load_prompt_template('./prompts/code_prompt_template.txt')
    max_tokens = 128
    temperature = 0
    # generate
    for question in tqdm(data):
        # print("Generating code for question: ", question['id'])
        prompt = prompt_template.format(question=question['question'])
        stop = ['</s>', 'def', '#', '\n\n']
        for i, sub in enumerate(question['sub_questions']):
            prompt += f"\n    # Q.{i+1}: " + sub
            output_text = call_no_interrupt(prompt, model, max_tokens, temperature, args.top_k, args.top_p, args.repetition_penalty, stop)
            
            # post-process output text for each step
            processed_output_text = post_process(output_text, stop)
            prompt += processed_output_text

            # early stop
            if 'return' in processed_output_text:
                break
        
        code = prompt[prompt.index('def q3():'):]

        # save to file
        generated_data.append(code)
    return generated_data
    
# def generate_no_decomp(filename: Path = 'decomposition_result.json'):
#     """
#     generate code for each question using IntelliCode approach
#     input: quesitons
#     output: generated Python code for all the quesitons
#     """
#     model = "togethercomputer/CodeLlama-13b-Python"

#     # read data
#     data = read_json(filename)
#     generated_data = []
#     # load prompt template
#     prompt_template = load_prompt_template('./prompts/code_no_decomp.txt')
#     max_tokens = 256
#     temperature = 0.2
#     # generate
#     for question in tqdm(data):
#         # print("Generating code for question: ", question['id'])
#         prompt = prompt_template.format(question=question['question'])
#         stop = ['</s>', 'def']
        
#         output_text = call_no_interrupt(prompt, model, max_tokens, temperature, args.top_k, args.top_p, args.repetition_penalty, stop)
#         processed_output_text = post_process(output_text, stop)
#         prompt += processed_output_text
#         code = prompt[prompt.index('def q3():'):]

#         # save to file
#         generated_data.append(code)
#     return generated_data

def execute(filename: Path = 'decomposition_result_with_code.json'):
    data = read_json(filename)
    generated_data = []
    def get_result(code):
        try:
            local = {}
            exec(code, globals(), local)
            q3 = local['q3']
            return q3()
        except:
            return 0
    for question in tqdm(data):
        print("Executing code for question: ", question['id'])
        if type(question['code']) == str:
            answer = get_result(question['code'])

        elif type(question['code']) == list:
            answer = []
            for code in question['code']:
                answer.append(get_result(code))
        generated_data.append(answer)
    
    return generated_data
    

def one_off(filename: Path):
    generated_code = generate(filename)
    questions = read_json(filename)
    for q, code in zip(questions, generated_code):
        q['code'] = code
    
    code_name = str(filename).replace('.json', '_with_code.json')
    with open(code_name, 'w') as f:
        json.dump(questions, f, indent=4)
    
    generated_answer = execute(code_name)
    for q, answer in zip(questions, generated_answer):
        q['answer'] = answer

    # save to file
    ans_name = str(filename).replace('.json', '_with_code_and_answer.json')
    with open(ans_name, 'w') as f:
        json.dump(questions, f, indent=4)


def majority_vote(filename: Path, num_votes = 10):
    agg_codes = []
    for i in range(num_votes):
        generated_code = generate(filename)
        agg_codes.append(generated_code)
    questions = read_json(filename)
    for i, (q, code) in enumerate(zip(questions, agg_codes)):
        q['code'] = [code[i] for code in agg_codes]
    
    code_name = str(filename).replace('.json', '_with_code.json')
    with open(code_name, 'w') as f:
        json.dump(questions, f, indent=4)
    
    generated_answer = execute(code_name)

    # get majority voted answer
    for i, (q, answer) in enumerate(zip(questions, generated_answer)):
        q['answer_votes'] = answer
        q['model_answer'] = statistics.mode(answer)
    # save to file
    ans_name = str(filename).replace('.json', '_with_code_and_answer.json')
    with open(ans_name, 'w') as f:
        json.dump(questions, f, indent=4)
    
    
if __name__ == "__main__":
    root = Path("./out")
    one_off(filename=root/'llama-2-13b-chat_gsm8k_decomp_planning_cot.json')
    # majority_vote()