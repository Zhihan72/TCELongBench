
import os
import pandas as pd
import json
import csv
from math import floor
from random import sample
import argparse
import openai
import tiktoken
import dateparse
from datetime import date
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer
from tensor_parallel import TensorParallelPreTrainedModel
from sentence_transformers import CrossEncoder
from fastchat.model import load_model, get_conversation_template, add_model_args

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "gpt-3.5-turbo-instruct"
encoding = tiktoken.encoding_for_model(model_engine)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dup = CrossEncoder('cross-encoder/quora-distilroberta-base',device=device)
model_sim = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
tokenizer_sim = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

def model_sim_similarity(sent, sent_list):
    similarity_list = []
    input_sent = tokenizer_sim(sent, padding=True, truncation=True, return_tensors="pt").to(device)
    inputs = tokenizer_sim(sent_list, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding_sent = model_sim(**input_sent, output_hidden_states=True, return_dict=True).pooler_output
        embeddings = model_sim(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    for embed in embeddings:
        cosine_sim = 1 - cosine(embedding_sent[0].cpu(), embed.cpu())
        similarity_list.append(cosine_sim)
    return similarity_list

def clean_each_point(str_):
    str_ = str_.strip().replace('* ','')
    for i in reversed(range(50)):
        str_ = str_.replace(f"{50-i}.",'').replace(f"{i+1},",'').replace(f"{i+1} ",'')
    return str_.strip()

##########################################
# Read Files of Articles, Complex Event, or Summary
##########################################
print('Read Files of Articles, Complex Event, or Summary')

f = open('./doc_filtered.json', 'r')
content = f.read()
GDELT_doc = json.loads(content)
f.close()

f = open('./TCE_ce_id.txt', 'r')
content = f.read()
ce_id_list = [int(item) for item in content.split(',')]
f.close()

summary_w_rules = pd.read_csv('./day_summary.csv')

num_tokens_input = []
num_tokens_output = []
save_path = './outline_points.csv'
with open(save_path, 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['ce_id', 'day', 'point', 'point_id', 'extracted_date', 'denoise_val', 'dup_loc', 'dup_val', 'sim_loc', 'sim_val', 'keep_val'])

    for ce_id_val in ce_id_list:
        print(f"\nCE ID: {ce_id_val}")
        # Summaries
        summary_ce_id_sample = summary_w_rules[summary_w_rules['ce_id']==ce_id_val]
        # Date List
        date_list = summary_ce_id_sample['day'].unique().tolist()
        # Md5 List
        Md5_list_ce_id_sample = summary_ce_id_sample['Md5_list'].unique().tolist()
        # Articles
        article_ce_id_sample = pd.DataFrame(data=None,columns=['ce_id', 'Md5', 'day', 'Title', 'Text'])
        flag = 0
        for idx in summary_ce_id_sample.index.tolist():
            date = summary_ce_id_sample.loc[idx]['day']
            Md5_list = summary_ce_id_sample.loc[idx]['Md5_list'].split(',')
            for Md5_value in Md5_list:
                article_ce_id_sample.loc[flag] = [ce_id_val, Md5_value, date, GDELT_doc[Md5_value]['Title'], GDELT_doc[Md5_value]['Text']]
                flag += 1

        ##########################################
        #Extract Points from Summary Using Prompts
        ##########################################
        print('Extract Points from Summary Using Prompts')

        outline_ce_id_sample = []
        for summary in summary_ce_id_sample['Summary'].tolist():
            msg = f"""
            You are an expert in extracting key contents from articles. Please extract the key points from the article with the following rules:
            1. The points should be independent from each other and have little overlaps.
            2. The content of the points should be concise, accurate and complete, especially for numbers, names and dates.
            3. If the point discusses the event happened over one month ago, you should discard it and only keep those discussing the event that just happened.
            4. Basically NO "he, she, they, it, them, etc" are allowed in the point. Please clearly write out the entity you are referencing in the point.
            5. You are not allowed to start the point with any of the following phrases: the article discusses, the article shows, the article emphasizes, the article discusses, the speaker says, the speaker discusses, the author mentions, etc.

            Here are several examples of extracting key points from articles. Note that the articles in different examples are irrelevant.

            Example 1:
            Article:
            Jordan took a firm stance against Israel after two Jordanians were killed in an apartment rented to an Israeli guard, demanding justice and an apology. The Israeli guard, Ziv Moyal, was angered by a dispute over furniture delivery and allegedly shot the two Jordanians, including a surgeon, Dr Bashar Hamarneh. Jordan communicated its demands through extensive channels, and Israel eventually issued an "apology memorandum" and agreed to legal action. The incident was a repeat of a previous attempt on the life of a Hamas leader in Amman, which was averted due to the intervention of the late King Hussain. The latest diplomatic breakthrough between Jordan and Israel was facilitated by the international media spin on the killings, which violated human values and scuttled Israeli attempts at image makeover.
            Key Points:
            * Jordan took a firm stance against Isreal after two Jordanians were killed in an apartment rented to an Israeli guard, and demand justice and an apology.
            * Israeli guard Ziv Moyal allegedly shot the two Jordanians.
            * Israel issued an "apology memorandum" and agreed to legal action about the accident caused by Israeli guard Ziv Moyal.

            Example 2:
            Article:
            Islamic Jihad has threatened military action against Israel if Palestinian prisoner Hisham Abu Hawash, who is on a hunger strike, dies. Abu Hawash has been on a hunger strike for more than four months in protest of his detention without trial. Islamic Jihad spokesman Daoud Shihab said that "all options are on the table" and that the group is in urgent contact with Egyptian mediators to prevent an escalation. Senior Islamic Jihad official Khaled al-Batash said that if Abu Hawash dies, there would be a joint response from all factions in Gaza, including Hamas' military wing. Dozens of protests and strikes are taking place in Palestinian cities in solidarity with Abu Hawash, including a planned strike on Tuesday in his hometown of Dura.
            Key Points:
            * Islamic Jihad has threatened military action against Israel if Palestinian prisoner Hisham Abu Hawash dies.
            * Islamic Jihad is in urgent contact with Egyptian mediators to prevent an escalation.
            * Islamic Jihad would start a joint response from all factions in Gaza, including Hamas' military wing if Palestinian prisoner Hisham Abu Hawash dies.
            * Protests and strikes take place in Palestinian cities in solidarity with Palestinian prisoner Hisham Abu Hawash.

            Example 3:
            Article:
            Israel has announced that it is gradually reopening its embassy in Jordan after a shutdown prompted by a deadly shooting in the embassy's vicinity last year. The shooting, which was carried out by a security guard for the Israeli embassy, resulted in the death of two Jordanian workers, including one who had stabbed the guard with a screwdriver. The incident sparked widespread anger in Jordan, and the Jordanian government refused to allow the embassy staff to return until Israel opened a serious investigation and offered an apology. In January, Israel reportedly apologized and agreed to compensate the families of the victims, and the conditions for reopening the embassy were met. The embassy staff received a hero's welcome from Israeli Prime Minister Benjamin Netanyahu, who was accompanied by the Israeli ambassador.
            Key Points:
            * Israel has announced to gradually reopen Isreal's embassy in Jordan after a shutdown.
            * One Jordanian worker stabbed a security guard for the Israeli embassy with a screwdriver, and the guard shot two Jordanian workers to death.
            * The Jordanian government refused to allow the security guard to return until Israel opened a serious investigation and offered an apology.
            * Israel reportedly apologized and agreed to compensate the families of the victims to meet the conditions for reopening the Israeli embassy in Jordan.
            * The security guard received a hero's welcome from Israeli Prime Minister Benjamin Netanyahu.

            Given the above rules and examples, please extract the key points of the following article and output them in the same way as examples.
            Article:
            {summary}
            Key Points:
            """
            response = ask_ChatGPT(msg)
            outline_ce_id_sample.append(response)
            # Record the tokens used
            num_tokens_input.append(len(encoding.encode(msg)))
            num_tokens_output.append(len(encoding.encode(response)))
            

        points_ce_id = []
        point_ids_ce_id = []
        day_list = []
        extracted_date_loc_list = []
        for i in range(len(outline_ce_id_sample)):
            point_list = [clean_each_point(item.strip()) for item in outline_ce_id_sample[i].split('\n') if item != ""]
            date = date_list[i]
            points_ce_id.extend(point_list)
            point_ids_ce_id.extend([j for j in range(len(point_list))])
            day_list.extend([date for j in range(len(point_list))])
            extracted_date_loc_list.extend([('',-1) for k in range(len(point_list))])

        ##########################################
        # Filter Out Redundant & Noising Points
        ##########################################
        print('Filter Out Redundant & Noising Points')

        denoise_list = []
        dup_list = []
        sim_list = []
        keep_list = []
        dup_loc_list = []
        sim_loc_list = []

        for idx in range(len(points_ce_id)):
            denoise_val = 0
            dup_val = 0
            sim_val = 0
            keep_val = 1
            dup_loc = [] # (day, point_id)
            sim_loc = [] # (day, point_id)
            point_idx = points_ce_id[idx]

            score_dup = model_dup.predict([(point_idx, item) for item in points_ce_id]).tolist()
            score_sim = model_sim_similarity(point_idx, points_ce_id)
            score_denoise = (sum(score_sim)-score_sim[idx]) / (len(score_sim)-1)      
            dup_loc = [(day_list[k], point_ids_ce_id[k]) for k in range(len(score_dup)) if score_dup[k] >= 0.8]
            sim_loc = [(day_list[k], point_ids_ce_id[k]) for k in range(len(score_sim)) if score_sim[k] >= 0.8]
            itself_loc = (day_list[idx], point_ids_ce_id[idx])
            if itself_loc in dup_loc: dup_loc.remove(itself_loc)
            if itself_loc in sim_loc: sim_loc.remove(itself_loc)

            denoise_val = 1 if score_denoise < 0.2 else 0
            for (day_i, id_i) in dup_loc:
                if day_i < day_list[idx]:
                    dup_val = 1
                    break
                elif day_i == day_list[idx]:
                    if id_i  < point_ids_ce_id[idx]:
                        dup_val = 1
                        break
            for (day_i, id_i) in sim_loc:
                if day_i < day_list[idx]:
                    sim_val = 1
                    break
                elif day_i == day_list[idx]:
                    if id_i < point_ids_ce_id[idx]:
                        sim_val = 1
                        break
            if denoise_val == 0 and dup_val == 0 and sim_val == 0:
                keep_val = 1
            else:
                keep_val = 0

            denoise_list.append(denoise_val)
            dup_list.append(dup_val)
            sim_list.append(sim_val)
            keep_list.append(keep_val)
            dup_loc_list.append(dup_loc)
            sim_loc_list.append(sim_loc)

        for idx in range(len(points_ce_id)):
            writer.writerow([ce_id_val, day_list[idx], points_ce_id[idx].strip(), point_ids_ce_id[idx], extracted_date_loc_list[idx], denoise_list[idx], dup_loc_list[idx], dup_list[idx], sim_loc_list[idx], sim_list[idx], keep_list[idx]])