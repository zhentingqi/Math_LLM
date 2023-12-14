import pdb
from API_call import call, call_no_interrupt
import re
import json
from tqdm import tqdm


#! naive decomposition
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


#! cot decomposition
decompose_prompt_with_cot = """
Given a question, please decompose it into sub-questions. When the original question is answerable, please start the subquestion with \"Now we can answer the question: \".

Question 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?
Let's think step by step. First, we need to find out how old is Mohamed. Next, we want to know how old was Mohamed four years ago. Then we can know how old was Kody four years ago. Finally, we can answer how old is Kody?
Question 1.1: How old is Mohamed?
Question 1.2: How old was Mohamed four years ago?
Question 1.3: How old was Kody four years ago?
Question 1.4: Now we can answer the question: How old is Kody? 

Question 2: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies before two of the fireflies flew away. How many fireflies remained?
Let's think step by step. First, we need to find out how many fireflies were there at first. Second, we want to know how many fireflies flew away from them. Third, we can know how many fireflies remained after they flew away. Finally, we can answer how many fireflies remained.
Question 2.1: How many fireflies were there at first?
Question 2.2: How many fireflies flew away from them?
Question 2.3: How many fireflies remained after they flew away?
Question 2.4: Now we can answer the question: How many fireflies remained?

Question 3: Ali has four $10 bills and six $20 bills that he saved after working for Mr. James on his farm. Ali gives her sister half of the total money he has and uses 3/5 of the remaining amount of money to buy dinner. Calculate the amount of money he has after buying the dinner.
Let's think step by step. First, we need to find out what is the total amount of money Ali has. Second, we want to know how much money Ali gave to his sister. Third, we can know how much money Ali has after giving his sister the money. Fourth, we can know how much money Ali used to buy the dinner. Finally, we can answer how much money Ali had after buying the dinner.
Question 3.1: What is the total amount of money Ali has?
Question 3.2: How much money Ali gave to his sister?
Question 3.3: How much money does Ali have after giving his sister the money?
Question 3.4: How much money Ali used to buy the dinner?
Question 3.5: Now we can answer the question: How much money Ali had after buying the dinner?

Question 4: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn?
Let's think step by step. First, we need to find out how far the car traveled after the 1st turn. Second, we want to know how far the car traveled after the 2nd turn. Finally, we can answer how far the car traveled after the 3rd turn.
Question 4.1: How far did the car travel after the 1st turn?
Question 4.2: How far did the car travel after the 2nd turn?
Question 4.3: Now we can answer the question: How far did the car travel after the 3rd turn?
"""


#! planning cot decomposition
decompose_prompt_with_planning_cot = """
Let's tackle each problem by first outlining our plan of action list. Then, we'll proceed step by step, explaining our logic at each stage and formulating a specific subquestion. Once all subproblems are addressed, we'll combine our findings to answer the original question. When the original question is answerable, please start the subquestion with \"Now we can answer the question: \".

Question 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?

Plan and Reasoning:
- First, we need to determine Mohamed's current age by doubling 30.
- Next, we figure out how old Mohamed was four years ago.
- Then, based on Mohamed's age four years ago, we calculate half of that age to determine Kody's age at that time.
- Finally, we'll add four years to Kody's age back then to find his current age.
- Now, we'll formulate our subproblems to address each step.

Subproblems:
Question 1.1: If Mohamed is twice as 30 years old, what is his current age?
Question 1.2: What was Mohamed's age four years ago?
Question 1.3: If Kody was half as old as Mohamed four years ago, what was Kody's age?
Question 1.4: Now we can answer the question: Adding four years to Kody's age back then, how old is Kody now?

Question 2: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies before two of the fireflies flew away. How many fireflies remained?

Plan and Reasoning:
- First, we have three fireflies.
- Next, we need to calculate a dozen minus four to determine how many more fireflies joined.
- Then, combining the initial fireflies with the newcomers, we subtract the two that flew away.
- Finally, we'll have the number of fireflies that remained.
- Now, let's translate this plan into subproblems.

Subproblems:
Question 2.1: What is a dozen minus four?
Question 2.2: How many fireflies were there after the new ones joined the initial three?
Question 2.3: How many fireflies flew away?
Question 2.4: Now we can answer the question: After two flew away, how many fireflies remained?

Question 3: Ali has four $10 bills and six $20 bills that he saved after working for Mr. James on his farm. Ali gives his sister half of the total money he has and uses 3/5 of the remaining amount of money to buy dinner. Calculate the amount of money he has after buying the dinner.

Plan and Reasoning:
- First, we calculate the total amount of money from the $10 and $20 bills.
- Next, we find out the amount Ali gives to his sister, which is half of his total.
- Then, we determine what remains after giving money to his sister.
- Then, we can calculate 3/5 of the remaining money that Ali uses to buy dinner.
- Finally, the remaining money after this purchase is what Ali has left.
- Now, we'll break this down into subproblems to solve sequentially.

Subproblems:
Question 3.1: How much money does Ali have in total from the $10 and $20 bills?
Question 3.2: How much money does Ali give to his sister?
Question 3.3: What amount does Ali have left after giving money to his sister?
Question 3.4: How much of this remaining amount does Ali use to buy dinner?
Question 3.5: Now we can answer the question: After buying dinner, how much money does Ali have left?

Question 4: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn?

Plan and Reasoning:
- First, we add the distance the car traveled after the 1st turn (5 meters) to the distance after the 2nd turn (8 meters).
- Next, we subtract the total distance traveled after the first two turns from the overall distance of 23 meters to find the combined distance for the 3rd and 4th turns.
- Then, considering the car exits immediately after the 4th turn, we assume there's no additional distance traveled at that point, which means the remaining distance pertains solely to the 3rd turn.
- Finally, we calculate the exact distance the car traveled after the 3rd turn.
- Now, we will formulate subproblems based on these steps to guide us to the solution sequentially.

Subproblems:
Question 4.1: If the car traveled 5 meters after the 1st turn, and 8 meters after the 2nd turn, what is the total distance traveled after these two turns?
Question 4.2: Given the car traveled a total of 23 meters, how much distance is left for the 3rd and 4th turns combined?
Question 4.3: If the car exits the tunnel immediately after the 4th turn without traveling further, how far did it travel after the 3rd turn?
Question 4.4: Now we can answer the question: How far did the car travel after the 3rd turn?
"""



def decompose(question: str, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop, type):
    count = 0
    def clear_question_marks(text):
        # Regex pattern to find all occurrences of "Question 5.x:"
        # \d+ matches one or more digits
        pattern = r'Question 5\.\d+:'
        
        # Using re.sub() to replace the matched patterns with an empty string
        cleaned_text = re.sub(pattern, '', text)
        
        return cleaned_text.strip()
    
    if type == "naive":
        prompt = decompose_prompt + "\n" + "Question 5: " + question + "\n" + "Question 5.1: "
    elif type == "cot":
        prompt = decompose_prompt_with_cot + "\n" + "Question 5: " + question + "\n" + "Let's think step by step. "
    elif type == "planning_cot":
        prompt = decompose_prompt_with_planning_cot + "\n" + "Question 5: " + question.strip() + "\n" + "\nPlan and Reasoning:\n"
    
    output_text = ""
    cnt = 0
    while "Now we can answer the question" not in output_text and cnt <= 10:
        output_text = call_no_interrupt(prompt, model, max_tokens, temperature, top_k, top_p, repetition_penalty, stop)
        cnt += 1

    if type == 'naive':
        truncated_output_text = output_text.split("Question 6")[0]
    elif type == "cot":
        truncated_output_text = output_text.split("Question 6")[0]
    elif type == "planning_cot":
        try:
            truncated_output_text = output_text.split("Subproblems:")[1].split("\n\n")[0]
        except:
            truncated_output_text = output_text.split("Subproblem:")[1].split("\n\n")[0]

    subquestions = []
    for line in truncated_output_text.split("\n"):
        if "Question 5." in line:
            sub_question = clear_question_marks(line)
            subquestions.append(sub_question)
    
    if len(subquestions) == 0:
        count += 1
        subquestions.append(question.strip())
    print(count)
    return subquestions


def decompose_all(model, dataset, type):
    from main import args
    model_name = model.split("/")[-1]   # e.g. model: togethercomputer/llama-2-70b-chat

    src = f"./data/{dataset}/test_with_ids.json"
    tgt = f"./out/{model_name}_{dataset}_decomp_{type}.json"

    with open(src, "r") as input_file, open(tgt, "w") as output_file:
        test_data = json.load(input_file)
        all_items = []
        for i, qa_pair in tqdm(enumerate(test_data), total=len(test_data)):
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
                type=type
            )

            qa_pair["sub_questions"] = subquestions

            all_items.append(qa_pair)

        json.dump(all_items, output_file)

if __name__ == "__main__":
    # decompose_all(model="togethercomputer/llama-2-7b-chat", dataset="SVAMP", type="naive")
    # decompose_all(model="togethercomputer/llama-2-7b-chat", dataset="SVAMP", type="cot")

    # TODO
    # decompose_all(model="togethercomputer/llama-2-7b-chat", dataset="SVAMP", type="planning_cot")
    decompose_all(model="togethercomputer/llama-2-13b-chat", dataset="SVAMP", type="naive")
    decompose_all(model="togethercomputer/llama-2-13b-chat", dataset="SVAMP", type="cot")

    # TODO
    # decompose_all(model="togethercomputer/llama-2-13b-chat", dataset="SVAMP", type="planning_cot")