import os
import argparse
import random
import math
import datetime
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
from Verify_Forecast_QA import forecastqa_post_verify

print('Loading models and tokenizers')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "gpt-3.5-turbo-instruct"
encoding = tiktoken.encoding_for_model(model_engine)
model_dup = CrossEncoder('cross-encoder/quora-distilroberta-base',device=device)

def ask_ChatGPT(msg):
    completion = openai.Completion.create(
        model=model_engine,
        prompt=msg,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = completion.choices[0].text
    return response

def str_to_list_id(str_):
    list_1 = str_.replace('[','').replace(']','').split(',')
    return [int(i.strip()) for i in list_1]

def cal_date_gap(dt_1, dt_2):
    year_1 = math.floor(dt_1*0.0001)
    year_2 = math.floor(dt_2*0.0001)
    month_1 = math.floor((dt_1 - year_1*10000)*0.01)
    month_2 = math.floor((dt_2 - year_2*10000)*0.01)
    day_1 = dt_1 - year_1*10000 - month_1*100
    day_2 = dt_2 - year_2*10000 - month_2*100
    d1 = datetime.datetime(year_1,month_1,day_1)
    d2 = datetime.datetime(year_2,month_2,day_2)
    interval = d2 - d1
    return interval.days

f = open('./doc_filtered.json', 'r')
content = f.read()
GDELT_doc = json.loads(content)
f.close()

day_summary = pd.read_csv("./day_summary.csv")
outline_points = pd.read_csv("./outline_points.csv")
ce_id_list = outline_points['ce_id'].unique().tolist()
print(f"# of CE included: {len(ce_id_list)}")

num_tokens_input = []
num_tokens_output = []
letter_dict = ['(a)','(b)','(c)','(d)']

save_path = './dataset/TLB_forecast_creation.csv'
with open(save_path, 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['ce_id', 'Md5_1', 'Md5_2', 'day_1', 'day_2', 'point', 'point_id', 'question', 'answer', 'evidence', 'choices', 'shuffle_order', 'ground_truth'])
    
    for ce_val_i in ce_id_list:
        print(f"CE id: {ce_val_i}")

        points_ce_value = outline_points[(outline_points['ce_id']==ce_val_i) & (outline_points['keep_val']==1)].sort_values(by = ['day','point_id'],ascending=[True,True])
        day_list_unique = points_ce_value['day'].unique().tolist()
        if len(day_list_unique) < 3:
            print(f"Skip ce id = {ce_val_i} for only {len(day_list_unique)} days!")
            continue
        day_last = day_list_unique[-1]
        day_last_str = '-'.join([str(day_last)[:4], str(day_last)[4:6], str(day_last)[6:]])
        print(f"Last day: {day_last_str}")
        
        df_points_day_last = outline_points[(outline_points['ce_id']==ce_val_i) & (outline_points['keep_val']==1) & (outline_points['day']==day_last)].sort_values(by = ['day','point_id'],ascending=[True,True])
        points_day_last = df_points_day_last['point'].tolist()
        points_id_day_last = df_points_day_last['point_id'].tolist()

        Md5_day_last = day_summary[(day_summary['ce_id']==ce_val_i)&(day_summary['day']==day_last)]['Md5_list'].tolist()[0]
        Md5_list_day_last = [item.strip() for item in Md5_day_last.split(',') if len(item.strip())!=0]
        Md5_last = random.sample(Md5_list_day_last, 1)[0]
        article_last = '\n'.join(GDELT_doc[Md5_last]['Text'])

        for point_k in points_day_last:
            flag = 1
            counter = 0
            point_id_k = points_id_day_last[points_day_last.index(point_k)]
            print(f"Point Now ({point_id_k}): {point_k}")

            while (flag > 0) and (counter < 5): # 5
                # Creating question-answer pair
                prompt_1 = f"""
                Imageine the scenario:
                * Today is {day_last_str}.
                * The article provided has just been published.

                Article:
                {article_last}
                Publishing date: {day_last_str}

                Please generate one forecasting question about the above article, along with its answer. You should follow the instructions below:
                1. The question should be around the key point "{point_k}" and come from the above article.
                2. The question must be guessable, but not answerable until {day_last_str}. 
                3. The question should start with one of the following phrases: "What will", "Who will", "Where will", "Which country will", "Why will", "How much", "How will", "How many".
                4. There must be a time element in the question. It can be phrases like "In {day_last_str} ...", "After {day_last_str}, ...", "... in {day_last_str}?". However, you are NOT allowed to use "before" in the question, as remember the question should be able to be answered without information from the day the article was published.
                5. You should avoid the following question types: questions that require numerical reasoning (this is not a math test); questions that require substantial world knowledge.
                6. The answer to the question MUST be short and concise, avoiding using redundant words or repeating the information in the question. There must be direct evidence to support the answer.
                7. The question must be grammatically correct and contain all the information required to answer. Basically NO "he, she, they, it, them, etc" are allowed in the question. Please clearly write out the entity you are referencing in the foercasting question.

                Here are some examples showing the writing style. NOTE that the content of the examples are irrelevant to the question you will generate.
                * Question: What will Belinda Carlisle want to be by 2019-09-01?\nAnswer: Travel Agent
                * Question: Who will visit Pittsburgh for first 2020 campaign rally in 2019-04-12?\nAnswer: Joe Biden
                * Question: Where will the Glasgow derby be played in 2021-05-01?\nAnswer: Scotland
                * Question: What will be M&S's response after their shares fall in 2016-03-24?\nAnswer: They will focus on the goal and aim to regenerate the business within the next 5 years
                * Question: What will Trump say that will happen to the economy if he's not reelected in 2017-08-13?\nAnswer: The economy will tank

                Now please write a question following the instructions and examples above. You should output the question along with its answer, in the format of "Question: xxx?\nAnswer: xxx.".
                """
                try:
                    response_1 = ask_ChatGPT(prompt_1)
                    num_tokens_input.append(len(encoding.encode(prompt_1)))
                    num_tokens_output.append(len(encoding.encode(response_1)))
                except:
                    print(f"Exceed the max length of ChatGPT (ce id = {ce_val_i}), last date = {day_last})!")
                    counter += 1
                    continue

                resp_list = [item.replace('Question:','').replace('Answer:','').strip() for item in response_1.split('\n') if len(item.strip())!=0]
                if len(resp_list)!=2:
                    print(f"Error processing response_1 (ce id = {ce_val_i}), last date = {day_last})!")
                    counter += 1
                    continue

                question_k = resp_list[0]
                answer_k = resp_list[1]
                print(f"ques: {question_k}")
                print(f"ans: {answer_k}")

                if day_last_str not in question_k:
                    print(f"Last day {day_last_str} is not in the question {question_k}.")
                    counter += 1
                    continue

                # Pre Verify
                prompt_2 = f"""
                Imageine the scenario:
                * Today is {day_last_str}.
                * The article provided have just been published.

                Article:
                {article_last}
                Publishing date: {day_last_str}
                Forecasting question: {question_k}

                Now imagine you go back to the "past" (any day before {day_last_str}) and ask the forecasting question above.

                Q1. Do you think there will be anybody (friend, family, stranger, anyone real) in the "past" who could make an educated guesses as to what the answer to the forecasting question is?
                A. Yes, there would be at least one person who could make an educated guess as to what the answer to the forecasting question is.
                B. No, there wouldn't be a single person who could make an educated guess as to what the answer to the forecasting question is.

                Q2. Do you think a few people (not including people mentioned in the article) in the "past" could answer the forecasting question with 100% certainty without you telling them information from {day_last_str} (the day article was published)?
                A. Yes, there could be a few people who could answer the forecasting question with 100% certainty without me telling them information from {day_last_str}.
                B. No, there wouldn't really be anyone who could answer the forecasting question without information from {day_last_str}.

                Please output your answer to Q1 and Q2, in the format of "Q1: x\nQ2: x".
                """
                try:
                    response_2 = ask_ChatGPT(prompt_2)
                    num_tokens_input.append(len(encoding.encode(prompt_2)))
                    num_tokens_output.append(len(encoding.encode(response_2)))
                except:
                    print(f"Exceed the max length of ChatGPT (ce id = {ce_val_i}), last date = {day_last})!")
                    counter += 1
                    continue
                resp_list_verify = [item.replace('Q1:','').replace('Q2:','').replace('.','').strip()[0] for item in response_2.split('\n') if len(item.strip())!=0]


                # Find out the evidence
                prompt_2_evi = f"""
                Article:
                {article_last}
                Question: {question_k}
                Answer: {answer_k}

                Given the above article, please check if the answer is correct to the question with 100% certainty. You should follow the instructions below:
                1. You should first find the relevant sentences from the above article.
                2. You should then reason out the answer to the above question step by step.
                3. Finally, you should compare your answer with the above one. 

                If the above answer is the same as the one you got, please output "The given answer is correct." along with one original sentence that supports the answer the most strongly; otherwise, output "The given answer may be wrong." along with one original sentence that rejects the answer the most strongly.
                """
                try:
                    response_2_evi = ask_ChatGPT(prompt_2_evi)
                    num_tokens_input.append(len(encoding.encode(prompt_2_evi)))
                    num_tokens_output.append(len(encoding.encode(response_2_evi)))
                except:
                    print(f"Exceed the max length of ChatGPT (ce id = {ce_val_i}), last date = {day_last})!")
                    counter += 1
                    continue

                evidence_i = response_2_evi.replace('\n','').replace('Evidence:','')
                evidence_bool = 1
                if "wrong" in response_2_evi:
                    evidence_bool = 0
                resp_list_verify.append(str(evidence_bool))

                if (resp_list_verify[0]!='A') or (resp_list_verify[1]!='B') or (resp_list_verify[2]!='1'):
                    print(f"Not pass the pre verifying phrase! Result is {','.join(resp_list_verify)}")
                    counter += 1
                    continue

                # Post verify
                post_verify = forecastqa_post_verify(question_k, answer_k, evidence_i, day_last_str)
                drop_bool_i = post_verify.compute_drop_bool()
                if (drop_bool_i==1):
                    print("In post verifying phrase, the forecasting characteristic of the question is not passed!")
                    counter += 1
                    continue
                num_tokens_input.append(400*3)
                num_tokens_output.append(15*3)

                flag_noi = 1
                counter_noi = 0
                while (flag_noi > 0) and (counter_noi < 3):
                    day_2 = random.sample(day_list_unique[:-1], 1)[0]
                    if day_2 not in day_summary[(day_summary['ce_id']==ce_val_i)]['day'].tolist():
                        counter_noi += 0.5
                        continue
                    Md5_day_2 = day_summary[(day_summary['ce_id']==ce_val_i)&(day_summary['day']==day_2)]['Md5_list'].tolist()[0]
                    Md5_list_day_2 = [item.strip() for item in Md5_day_2.split(',') if len(item.strip())!=0]
                    Md5_2 = random.sample(Md5_list_day_2,1)[0]
                    article_day_2 = '\n'.join(GDELT_doc[Md5_2]['Text'])
                    print(f"Random day: {day_2}")

                    # Creat noising answers
                    prompt_3 = f"""
                    Background 1:
                    {article_last}
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
                    What will Angela Merkel's government agree to support a $60 billion package for in September 2019?
                    Correct Answer:
                    Climate Polices
                    Nosing Answers:
                    (b) Infrastructure
                    (c) Immigration polices
                    (d) Health care

                    Question:
                    What will Belinda Carlisle want to be by 2019-09-01?
                    Correct Answer:
                    Pop star
                    Nosing Answers:
                    (b) Traveler
                    (c) Travel Agent
                    (d) American

                    Question:
                    Where will the Glasgow derby be played in 2021-05-01?
                    Correct Answer:
                    Wales
                    Nosing Answers:
                    (b) Scotland
                    (c) England
                    (d) Ireland

                    Question:
                    What will be M&S's response after their shares fall in 2016-03-24?
                    Correct Answer:
                    They will focus on the goal and aim to regenerate the business within the next 5 years
                    Nosing Answers:
                    (b) They will liquidate all stocks and close
                    (c) They will focus on the scoreboard and increase their blue-chip index
                    (d) They will fire 10 employees from each store

                    Question:
                    What will Trump say that will happen to the economy if he's not reelected in 2017-08-13?
                    Correct Answer:
                    The economy will tank
                    Nosing Answers:
                    (b) He will not leave power
                    (c) The country will go to war
                    (d) He will run again 4 years later

                    Now please generate three noising answers to the question "{question_k}" based on above instructions. DO NOT output any other explanations.

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
                        print(f"Exceed the max length of ChatGPT (ce id = {ce_val_i}, last date = {day_last})!")
                        counter_noi += 1
                        continue

                    choices_resp = '(a) '+ answer_k +'\n' + response_3
                    four_choices = [ch.replace('(a)','').replace('(b)','').replace('(c)','').replace('(d)','').strip() for ch in choices_resp.split('\n') if len(ch.strip())>0]
                    if (len(list(set(four_choices))) != 4):
                        print('Wrong output format! (prompt_2)')
                        #print(f"Not saved. {four_choices}")
                        counter_noi += 1
                        continue
                    print(f"Choices: {four_choices}")
                    
                    # Check if the answers are duplicated or do not share the similar wording
                    len_four_choices = [len(item.split(' ')) for item in four_choices]
                    if (max(len_four_choices) - min(len_four_choices) > 10):
                        counter_noi += 1
                        print("The length of four choices is not similar.")
                        continue
                    dup_flag = 0                    
                    for choice_j in four_choices:
                        score_dup = model_dup.predict([(choice_j, item) for item in four_choices if item!=choice_j]).tolist()
                        if max(score_dup) > 0.8:
                            dup_flag = 1
                            break
                    if dup_flag == 1:
                        counter_noi += 1
                        print("There are answers that duplicate the other ones.")
                        continue

                    # Save the qualified question and four answers
                    num_random = random.sample(range(4), 4)
                    choices_i = [four_choices[id_] for id_ in num_random]
                    shuffle_order_i = [letter_dict[id_] for id_ in num_random]
                    ground_truth_i = chr(65+shuffle_order_i.index('(a)'))
                    choices_str_i = '\n'.join([chr(65+i)+'. '+choices_i[i] for i in range(len(choices_i))])
                    
                    flag_noi -= 1
                    flag -= 1
                    writer.writerow([ce_val_i, Md5_last, Md5_2, day_last, day_2, point_k, point_id_k, question_k, four_choices[0], evidence_i, choices_str_i, ','.join(shuffle_order_i), ground_truth_i])