import json

"""

Unifying the format of the json files

"""

with open("out/llama-2-7b-chat_gsm8k_decomp.json", 'r') as f:
    data = json.load(f)
for d in data:
    d['final_ans'] = d['answer']
    del d['answer']

with open("out/llama-2-7b-chat_gsm8k_decomp.json", "w") as f:
    json.dump(data, f, indent=4)

with open("out/one_off_llama-2-7b-chat_code34B_gsm8k_decomp_with_code.json", 'r') as f:
    data = json.load(f)
for d in data:
    d['final_ans'] = d['answer']
    del d['answer']

with open("out/one_off_llama-2-7b-chat_code34B_gsm8k_decomp_with_code.json", "w") as f:
    json.dump(data, f, indent=4)
    
with open("out/one_off_llama-2-7b-chat_code34B_gsm8k_decomp_with_code_and_answer.json", 'r') as f:
    data = json.load(f)
with open("out/one_off_llama-2-7b-chat_code34B_gsm8k_decomp_with_code.json", 'r') as f:
    ans_data = json.load(f)
for d, a in zip(data, ans_data):
    d['final_ans'] = a['final_ans']

with open("out/one_off_llama-2-7b-chat_code34B_gsm8k_decomp_with_code_and_answer.json", "w") as f:
    json.dump(data, f, indent=4)