src = "../one_off_decomposition_result_with_code_and_answer.json"

import json

num_correct = 0
num_tot = 0
with open(src, "r") as f:
    res = json.load(f)
    for item in res:
        gt = item["final_ans"]
        out = item["answer"]
    
        if out is not None and float(gt) == float(out):
            num_correct += 1
        
        num_tot += 1

print(num_correct / num_tot)
