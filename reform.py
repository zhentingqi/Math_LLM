import json
with open("out/llama-2-13b-chat_gsm8k_decomp_with_code_and_answer.json", 'r') as f:
    data = json.load(f)
with open("out/llama-2-13b-chat_gsm8k_decomp_with_code.json", 'r') as f:
    ans_data = json.load(f)
for d, a in zip(data, ans_data):
    d['final_ans'] = a['final_ans']

with open("out/llama-2-13b-chat_gsm8k_decomp_with_code_and_answer.json", "w") as f:
    json.dump(data, f, indent=4)