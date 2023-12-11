from decomposition import decompose_all
from IntelliCode import one_off
from pathlib import Path

if __name__ == "__main__":
    models = ["togethercomputer/llama-2-7b-chat", "togethercomputer/llama-2-13b-chat"]
    types = ["cot", "planning_cot"]
    for model in models:
        for typeee in types:
            try:
                decompose_all(model=model, dataset="gsm8k", type=typeee)
            except:
                print("model: ", model)
                print("method: ", typeee)
                continue
    
    datasets = ["llama-2-7b-chat_gsm8k_decomp_planning_cot.json",
                "llama-2-7b-chat_gsm8k_decomp_cot.json",
                "llama-2-13b-chat_gsm8k_decomp_planning_cot.json",
                "llama-2-13b-chat_gsm8k_decomp_cot.json"]
    for dataset in datasets:
        root = Path("./out")
        one_off(filename=root/dataset)
    


