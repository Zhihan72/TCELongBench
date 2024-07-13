import os
import argparse
import random
import json
import csv
import pandas as pd
import numpy as np
import openai
import tiktoken
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer
from tensor_parallel import TensorParallelPreTrainedModel
from fastchat.model import load_model, get_conversation_template, add_model_args

print('Loading models and tokenizers')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "gpt-3.5-turbo-instruct"
encoding = tiktoken.encoding_for_model(model_engine)
model_dup = CrossEncoder('cross-encoder/quora-distilroberta-base',device=device)

def ask_ChatGPT(msg, numtoken = 512):
    completion = openai.Completion.create(
        model=model_engine,
        prompt=msg,
        max_tokens= numtoken,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = completion.choices[0].text
    return response

def str_to_list_id(str_):
    list_1 = str_.replace('[','').replace(']','').split(',')
    return [int(i.strip()) for i in list_1]

def get_num_of_tokens(str_):
    return len(encoding.encode(str_))

f = open('./doc_filtered.json', 'r')
content = f.read()
GDELT_doc = json.loads(content)
f.close()

day_summary = pd.read_csv('./day_summary.csv')
outline_points = pd.read_csv("./outline_points.csv")
ce_id_list = outline_points['ce_id'].unique().tolist()

num_tokens_input = []
num_tokens_output = []
letter_dict = ['(a)','(b)','(c)','(d)']

save_path = './dataset/TLB_detail_creation.csv'
with open(save_path, 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['ce_id', 'Md5_1', 'Md5_2', 'day_1', 'day_2', 'point', 'point_id', 'question', 'answer', 'evidence', 'evi_bool', 'choices', 'shuffle_order', 'ground_truth'])

    for ce_val_i in ce_id_list:
        print(f"CE ID: {ce_val_i}\n")

        points_ce_value = outline_points[(outline_points['ce_id']==ce_val_i) & (outline_points['keep_val']==1) & (outline_points['point_id']!=-1)].sort_values(by = ['day','point_id'],ascending=[True,True])
        day_list_unique = points_ce_value['day'].unique().tolist()
        day_list_unique_woLast = day_list_unique[:-1]
        if len(day_list_unique_woLast) < 2:
            print(f"Skip ce id = {ce_val_i} for only {len(day_list_unique_woLast)} days except the last day!")
            continue

        # Propose questions each day in this complex event
        for day_1 in day_list_unique_woLast:

            # Get the points on day_1 and the Md5 values on day_1 and day_2
            day_list_rest = [item for item in day_list_unique_woLast if item!=day_1]
            df_points_day_1 = outline_points[(outline_points['ce_id']==ce_val_i) & (outline_points['keep_val']==1) & (outline_points['day']==day_1)].sort_values(by = ['day','point_id'],ascending=[True,True])
            points_day_1 = df_points_day_1['point'].tolist()
            points_id_day_1 = df_points_day_1['point_id'].tolist()
            Md5_day_1 = day_summary[(day_summary['ce_id']==ce_val_i)&(day_summary['day']==day_1)]['Md5_list'].tolist()[0]
            Md5_list_day_1 = [item.strip() for item in Md5_day_1.split(',') if len(item.strip())!=0]
            
            for point_k in points_day_1:
                flag = 1
                counter = 0
                point_id_k = points_id_day_1[points_day_1.index(point_k)]
                print(f"Point now ({point_id_k}): {point_k}")
                while (flag > 0) and (counter < 3): # 5
                    
                    # Randomly select another day to create noising answers
                    day_2 = random.sample(day_list_rest, 1)[0]
                    Md5_day_2 = day_summary[(day_summary['ce_id']==ce_val_i)&(day_summary['day']==day_2)]['Md5_list'].tolist()[0]
                    Md5_list_day_2 = [item.strip() for item in Md5_day_2.split(',') if len(item.strip())!=0]

                    Md5_1 = random.sample(Md5_list_day_1,1)[0]
                    Md5_2 = random.sample(Md5_list_day_2,1)[0]
                    article_day_1 = '\n'.join(GDELT_doc[Md5_1]['Text'])
                    article_day_2 = '\n'.join(GDELT_doc[Md5_2]['Text'])

                    # Propose question and its answer
                    prompt_1 = f"""
                    Article:
                    {article_day_1}

                    Given the above article, please generate one question along with its answer. You should follow the instructions below:
                    1. The question should be about the key point "{point_k}" and come from the above article.
                    2. The question should be unambiguous and challenging, avoiding simple string matching. There are NOT allowed to contain any sub-questions.
                    3. The question should be answerable based only on the text of the above article.
                    4. You should avoid the following question types: questions that require numerical reasoning (this is not a math test); questions that require substantial world knowledge; questions that require the reader to speculate.
                    5. The answer to the question MUST be short and concise, avoiding using redundant words or repeating the information in the question.
                    6. You should output the question and its answer without any other explanation, such as "Question: xxx?\nAnswer: xxx."
                    
                    Here are some examples showing the writing style. NOTE that the content of the examples are irrelevant to the question you will generate.
                    * Question: What does Holger von Neuhoff say about the bottled message?\nAnswer: It is the oldest message found along with the bottle he has ever encountered
                    * Question: Who first stated that the polygraph might not be reliable??\nAnswer: The psychologist William Martson
                    * Question: Where did Richard Platz want the postcard to end up?\nAnswer: At a museum
                    * Question: When are police stations expected to start using the new lie detection method?\nAnswer: Once it reaches an accuracy of at least 70%
                    * Question: What is a challenge working children face in regards to attending school, according to al-Mamun?\nAnswer: It can be hard for them to assimilate to the school environment

                    Now please write a question following the instructions and examples above. You should output the question along with its answer, in the format of "Question: xxx?\nAnswer: xxx.". NOTE that the answer should be as short as possible.
                    """
                    try:
                        response_1 = ask_ChatGPT(prompt_1)
                        num_tokens_input.append(len(encoding.encode(prompt_1)))
                        num_tokens_output.append(len(encoding.encode(response_1)))
                    except:
                        print('ChatGPT context length restriction! (prompt_1)')
                        counter += 1
                        continue
                    resp_list = [i.strip() for i in response_1.split('\n') if len(i.strip())>0]
                    if len(resp_list) != 2:
                        print('Wrong output format!(prompt_1)')
                        counter += 1
                        continue

                    question_k = resp_list[0].replace('Question:','').strip()
                    answer_k = resp_list[1].replace('Answer:','').strip()
                    print(f"ques: {question_k}")
                    print(f"ans: {answer_k}")

                    # Find the evidence to see if the answer is correct
                    prompt_2_evi = f"""
                    Article:
                    {article_day_1}

                    Question: {question_k}
                    Answer: {answer_k}

                    Given the above articles, please check if the answer is correct to the question with 100% certainty. You should follow the instructions below:
                    1. You should first find the relevant sentences from the above article.
                    2. You should then reason out the answer to the above question step by step.
                    3. Finally, you should compare your answer with the above one. 
                    
                    If the above answer is the same as the one you got, please output "The given answer is correct." along with one original sentence that supports the answer the most strongly; otherwise, output "The given answer may be wrong." along with one original sentence that rejects the answer the most strongly.
                    """
                    response_2_evi = ask_ChatGPT(prompt_2_evi)
                    evidence_i = response_2_evi.replace('\n','').replace('Evidence:','')
                    evidence_bool = 1
                    if "wrong" in response_2_evi:
                        evidence_bool = 0
                    num_tokens_input.append(len(encoding.encode(prompt_2_evi)))
                    num_tokens_output.append(len(encoding.encode(response_2_evi)))

                    # Generate nosing answers
                    prompt_3 = f"""
                    Background 1:
                    {article_day_1}
                    Background 2:
                    {article_day_2}

                    Given the above two backgrounds, please generate three noising answers to the question "{question_k}", whose correct answer is "{answer_k}". 
                    Name the three noising answers as (b), (c) and (d) respectively. You should follow the instructions below:
                    1. (b), (c) and (d) must share the similar wording and length with the correct answer "{answer_k}".
                    2. The four answers must be essentially different and contradictory. 
                    3. Answer (b) is incorrect and reflects a misunderstanding of Background 1. (b) should not repeat the information in the correct answer "{answer_k}".
                    4. Answer (c) is incorrect and comes from Background 2.
                    5. Answer (d) is incorrect and has no support in neither of the backgrounds. (d) may refer to general world knowledge.
                    6. While (c) and (d) should all be unambiguously incorrect, they should also make sense and be plausible answers to the question.
                    7. (c) and in some cases (b) could be correct (in part or fully) as a fact but not correct as an answer to the question. It's also fine for (c) to be an incorrect fact as long as it has textual support in Background 2.

                    Here are examples showing the output format. This example is NOT related to the noising answers you will generate.

                    Question:
                    Who threw the bottle into the Baltic Sea?
                    Correct Answer:
                    Angela Erdmann.
                    Nosing Answers:
                    (b) Angela Erdmann's grandfather.
                    (c) A museum worker.
                    (d) A fisherman.

                    Question:
                    What does Erdmann want to add to the bottle exhibit?
                    Correct Answer:
                    Pictures of the bottled message's author
                    Nosing Answers:
                    (b) A deciphered copy of the text
                    (c) A photo that depicts a young man throwing a bottle into the sea
                    (d) Excerpts from a book written by her grandfather

                    Question:
                    Where does Dunamn believe the athletic abilities of adults are derived from?
                    Correct Answer:
                    The month in which they were born in
                    Nosing Answers:
                    (b) The opportunities offered by UK Sport during their youth
                    (c) Primarily from their innate genetics
                    (d) A combination of multiple different factors

                    Question:
                    What is a challenge working children face in regards to attending school, according to al-Mamun?
                    Correct Answer:
                    It can be hard for them to assimilate to the school environment
                    Nosing Answers:
                    (b) After they stop working, they miss their friends from the factory
                    (c) SOHAY's classes are intended for parents and employers, not children
                    (d) They don't have enough preparation for the level of learning

                    Question:
                    When are police stations expected to start using the new lie detection method?
                    Correct Answer:
                    Once it reaches an accuracy of at least 70%
                    Nosing Answers:
                    (b) Within 10 years
                    (c) Once it is able to track the movements of the entire body
                    (d) It is already in use in many police stations
                    
                    Now please generate three noising answers to the question, given the above backgrounds, instructions and examples. DO NOT output the backgrounds, the question or any other explanations.

                    Question:
                    {question_k}
                    Correct Answer:
                    {answer_k}
                    Nosing Answers:
                    """
                    try:
                        response_3 = ask_ChatGPT(prompt_3)
                        num_tokens_input.append(len(encoding.encode(prompt_3)))
                        num_tokens_output.append(len(encoding.encode(response_3)))
                    except:
                        print('ChatGPT context length restriction! (prompt_3)')
                        counter += 1
                        continue

                    choices_resp = '(a) '+ answer_k +'\n' + response_3
                    four_choices = [ch.replace('(a)','').replace('(b)','').replace('(c)','').replace('(d)','').strip() for ch in choices_resp.split('\n') if len(ch.strip())>0]
                    if (len(list(set(four_choices))) != 4):
                        print('Wrong output format! (prompt_2)')
                        counter += 1
                        continue
                    print(f"Choices: {four_choices}")
                    
                    # Check if the answers are duplicated or do not share the similar wording
                    ## Length comparesion, continue if one of them is too long or too short
                    len_four_choices = [len(item.split(' ')) for item in four_choices]
                    if (max(len_four_choices) - min(len_four_choices) > 10) or (max(len_four_choices) > 20):
                        counter += 1
                        print("The length of four choices is not similar.")
                        continue
                    ## Duplication Score, continue if one of them exceed 0.8
                    dup_flag = 0                    
                    for choice_j in four_choices:
                        score_dup = model_dup.predict([(choice_j, item) for item in four_choices if item!=choice_j]).tolist()
                        if max(score_dup) > 0.8:
                            dup_flag = 1
                            break
                    if dup_flag == 1:
                        counter += 1
                        print("There are answers that duplicate the other ones.")
                        continue

                    num_random = random.sample(range(4), 4)
                    choices_i = [four_choices[id_] for id_ in num_random]
                    shuffle_order_i = [letter_dict[id_] for id_ in num_random]
                    ground_truth_i = chr(65 + shuffle_order_i.index('(a)'))
                    choices_str_i = '\n'.join([chr(65+i)+'. '+choices_i[i] for i in range(len(choices_i))])

                    flag -= 1
                    writer.writerow([ce_val_i, Md5_1, Md5_2, day_1, day_2, point_k, point_id_k, question_k, four_choices[0], evidence_i, evidence_bool, choices_str_i, ','.join(shuffle_order_i), ground_truth_i])