import json
from utils import read_json,load_prompt_template
from utils import get_args
from pathlib import Path
from API_call import call, call_no_interrupt
from tqdm import tqdm
from argparse import ArgumentParser


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


def generate(args, model: str, filename: Path = 'decomposition_result.json'):
    """
    generate code for each decomposed subquestion using IntelliCode approach
    input: quesitons with decomposed sub-questions
    output: generated Python code for all the quesitons
    """
    # read data
    data = read_json(filename)
    generated_data = []
    # load prompt template
    prompt_template = load_prompt_template('./prompts/2-shot_code_prompt_template.txt')
    max_tokens = 256
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
    

def one_off(args, model: str, dataset: Path):
    generated_code = generate(args, model, dataset)
    questions = read_json(dataset)
    for q, code in zip(questions, generated_code):
        q['code'] = code
    
    filename = str(dataset).split('/')
    code_model_name_mapping = {
        "togethercomputer/CodeLlama-34b-Python": "code34B",
        "togethercomputer/CodeLlama-13b-Python": "code13B",
    }
    model_name = code_model_name_mapping[model]
    print(filename)
    mid = filename[-1].replace(".json", "")
    code_name = f"./{filename[0]}/IntelliCode/one_off_{mid}_{model_name}_with_code.json"
    
    with open(code_name, 'w') as f:
        json.dump(questions, f, indent=4)
    
    generated_answer = execute(code_name)
    for q, answer in zip(questions, generated_answer):
        q['model_answer'] = answer

    # save to file
    ans_name = f"./{filename[0]}/IntelliCode/one_off_{mid}_{model_name}_with_code_and_answer.json"
    with open(ans_name, 'w') as f:
        json.dump(questions, f, indent=4)
    
    
if __name__ == "__main__":
    args = get_args()
    root = Path("./out/decomp_result")
    parser = ArgumentParser()
    
    # add model argument
    parser.add_argument('--analyst_model', type=str, default="togethercomputer/llama-2-7b-chat", help='analyst model')
    parser.add_argument('--solver_model', type=str, default="togethercomputer/CodeLlama-34b-Python", help='solver model')
    parser.add_argument('--dataset', type=str, default="multiarith",
                        choices=['multiarith', 'gsm8k', 'SVAMP'] , help='dataset')

    exec_args = parser.parse_args()
    dataset = root / (exec_args.analyst_model.split('/')[-1] + '_' + exec_args.dataset + '_decomp_cot.json')

    one_off(args=args, model=exec_args.solver_model, dataset=dataset)

    

