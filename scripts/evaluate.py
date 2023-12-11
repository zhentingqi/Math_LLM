sources = ['out/llama-2-7b-chat_gsm8k_decomp_cot_with_code_and_answer.json']

import json

for src in sources:
    print(src)
    num_correct = 0
    num_tot = 0
    with open(src, "r") as f:
        res = json.load(f)
        for item in res:
            gt = item['answer']
            out = item['model_answer']
    
            if "gsm8k" in src or 'result_test_with_ids_llama' in src:
                gt = gt.split("#### ")[-1]
            try:
                if abs(float(gt) - float(out)) <= 1e-5:
                    num_correct += 1
            except:
                num_correct += 0
            
            num_tot += 1

    print(num_correct / num_tot)
    print(" ")