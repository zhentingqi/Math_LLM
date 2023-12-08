import pdb
from API_call import call, call_no_interrupt
import re
import json
from tqdm import tqdm


decompose_prompt = """
Given a question, please decompose it into sub-questions. When the original question is answerable, please start the subquestion with \"Now we can answer the question: \".

Question 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?
Question 1.1: How old is Mohamed?
Question 1.2: How old was Mohamed four years ago?
Question 1.3: How old was Kody four years ago?
Question 1.4: Now we can answer the question: How old is Kody?

Question 2: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies before two of the fireflies flew away. How many fireflies remained?
Question 2.1: How many fireflies joined?
Question 2.2: Now we can answer the question: How many fireflies remained?

Question 3: Ali has four $10 bills and six $20 bills that he saved after working for Mr. James on his farm. Ali gives her sister half of the total money he has and uses 3/5 of the remaining amount of money to buy dinner. Calculate the amount of money he has after buying the dinner.
Question 3.1: How much money does Ali have in total?
Question 3.2: How much money does Ali give to his sister?
Question 3.3: How much money does Ali have after giving his sister the money?
Question 3.4: How much money does Ali use to buy dinner?
Question 3.5: Now we can answer the question: How much money does Ali have after buying the dinner?

Question 4: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn?
Question 4.1: How far did the car travel except for the 3rd turn?
Question 4.2: Now we can answer the question: How far did the car have to travel after the 3rd turn?
"""


def decompose(question: str, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop):
    def clear_question_marks(text):
        # Regex pattern to find all occurrences of "Question 5.x:"
        # \d+ matches one or more digits
        pattern = r'Question 5\.\d+:'
        
        # Using re.sub() to replace the matched patterns with an empty string
        cleaned_text = re.sub(pattern, '', text)
        
        return cleaned_text.strip()
    
    prompt = decompose_prompt + "\n" + "Question 5: " + question + "\n" + "Question 5.1: "
    
    output_text = ""
    cnt = 0
    while "Now we can answer the question" not in output_text and cnt <= 10:
        output_text = call_no_interrupt(prompt, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop)
        cnt += 1
    truncated_output_text = output_text.split("\n\n")[0]
    subquestions = []
    for line in truncated_output_text.split("\n"):
        sub_question = clear_question_marks(line)
        subquestions.append(sub_question)
    
    return subquestions


def _test():
    from main import args

    model = "togethercomputer/llama-2-13b-chat"
    dataset = "multiarith"

    src = f"./data/{dataset}/test_with_ids.json"
    tgt = f"./{model}_{dataset}_decomp.json"

    with open(src, "r") as input_file, open(tgt, "w") as output_file:
        multiarith_test = json.load(input_file)
        all_items = []
        for qa_pair in tqdm(multiarith_test):
            question = qa_pair["question"]

            subquestions = decompose(
                question=question,
                model=model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=args.stop,
            )

            qa_pair["sub_questions"] = subquestions

            all_items.append(qa_pair)

        json.dump(all_items, output_file)


if __name__ == "__main__":
    _test()
