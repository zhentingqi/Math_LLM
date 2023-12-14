sources = [
           'out/result_SVAMP_llama-2-70b-chat_4-shot.json',
           ]

import json

for src in sources:
    print(src)
    num_correct = 0
    num_tot = 0
    with open(src, "r") as f:
        res = json.load(f)
        for item in res:
            gt = item['final_ans'] if 'multiarith' in src else item['answer']
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