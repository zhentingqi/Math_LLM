# src = "out/result_llama-2-7b-chat_multiarith_decomp_llama-2-7b-chat_direct.json"
# src = "out/result_llama-2-7b-chat_gsm8k_decomp_llama-2-7b-chat_direct.json"

# sources = ["out/result_llama-2-7b-chat_gsm8k_decomp_llama-2-7b-chat_mathreg.json",
#            "out/result_llama-2-7b-chat_multiarith_decomp_llama-2-7b-chat_mathreg.json",
#            "out/result_llama-2-7b-chat_gsm8k_decomp_llama-2-13b-chat_mathreg.json",
#            "out/result_llama-2-7b-chat_multiarith_decomp_llama-2-13b-chat_mathreg.json",
#            "out/result_llama-2-13b-chat_gsm8k_decomp_llama-2-7b-chat_mathreg.json",
#            "out/result_llama-2-13b-chat_multiarith_decomp_llama-2-7b-chat_mathreg.json",
#            "out/result_llama-2-13b-chat_gsm8k_decomp_llama-2-13b-chat_mathreg.json",
#            "out/result_llama-2-13b-chat_multiarith_decomp_llama-2-13b-chat_mathreg.json"]

sources = ["out/llama-2-7b-chat_multiarith_decomp_planning_cot_with_code_and_answer.json"]

# sources = ["out/result_test_llama-2-7b-chat_fewshot.json",
#            "out/result_test_llama-2-7b-chat_zeroshot.json",
#            "out/result_test_llama-2-13b-chat_fewshot.json",
#            "out/result_test_llama-2-13b-chat_zeroshot.json",
#            "out/result_test_with_ids_llama-2-7b-chat_fewshot.json",
#            "out/result_test_with_ids_llama-2-7b-chat_zeroshot.json",
#            "out/result_test_with_ids_llama-2-13b-chat_fewshot.json"]
import json

eval_mode = True
for src in sources:
    print(src)
    if 'shot' in src:
        # eval mode for the baseline methods
        eval_mode = False
    num_correct = 0
    num_tot = 0
    with open(src, "r") as f:
        res = json.load(f)
        for item in res:
            if eval_mode:
                gt = item["final_ans"]
                out = item["answer"]
            elif eval_mode == False:
                # gsm8k
                if 'result_test_with_ids_llama' in src:
                    gt = item['answer']
                    out = item['model_answer']
                # multiarith
                elif 'result_test_llama' in src:
                    gt = item['final_ans']
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
