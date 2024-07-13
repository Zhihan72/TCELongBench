import os
import argparse
import random
import json
import csv
import math
import spacy
import pandas as pd
import numpy as np
import openai
import tiktoken
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer
from tensor_parallel import TensorParallelPreTrainedModel
from fastchat.model import load_model, get_conversation_template, add_model_args
from Verify_Order_QA import orderqa_post_verify

######################################
##### Loading Dataset and Model ######
######################################

print('Loading models and tokenizers')

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "gpt-3.5-turbo-instruct"
encoding = tiktoken.encoding_for_model(model_engine)

def ask_ChatGPT(msg, numtoken = 512):
    completion = openai.Completion.create(
        model=model_engine,
        prompt=msg,
        max_tokens=numtoken,
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

def str_to_list(str_):
    if str_ == '[]':
        return []
    else:
        results = []
        str_list = str_.split('),')
        for item in str_list:
            list_1 = item.strip().replace('[','').replace(']','').replace('(','').replace(')','').split(', ')
            results.append((int(list_1[0]),int(list_1[1])))
        return results

def str_to_list_loc(str_):
    Md5_value, id_ = str_.split(', ')
    Md5_value = Md5_value.strip().replace('(','').replace("'",'')
    id_ = int(id_.strip().replace(')',''))
    return (Md5_value, id_)

def clean_each_point(str_):
    str_ = str_.strip().replace('* ','')
    for i in range(50):
        str_ = str_.replace(f"{50-i}.",'').replace(f"{i+1},",'').replace(f"{i+1} ",'')
    return str_.strip()

def random_choices(points_list):
    random_numbers = random.sample(range(len(points_list)), len(points_list))
    ground_truth = [chr(65+i) for i in random_numbers]
    points_w_serial_numbers = [ground_truth[k] + '. ' + clean_each_point(points_list[k]) for k in range(len(points_list))]
    str_unordered = '\n'.join([points_w_serial_numbers[random_numbers.index(id_)] for id_ in range(len(points_list))])
    return (str_unordered, ground_truth)

outline_points = pd.read_csv("./outline_points.csv")
ce_id_list = outline_points['ce_id'].unique().tolist()
num_tokens_input = []
num_tokens_output = []

nlp = spacy.load("en_core_web_sm")
ent_type = ['EVENT', 'GPE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON']

save_path = './dataset/TLB_order_creation.csv'
with open(save_path, 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['ce_id', 'common_ent', 'points_id', 'points', 'day', 'choices', 'ground_truth'])

    for ce_val_i in ce_id_list:
        print(f"CE id: {ce_val_i}\n")

        day_list_i = outline_points[(outline_points['ce_id']==ce_val_i) & (outline_points['keep_val']==1)]['day'].unique().tolist()
        day_last_i = sorted(day_list_i)[-1]
        points_ce_value = outline_points[(outline_points['ce_id']==ce_val_i) & (outline_points['keep_val']==1) & (outline_points['day']!=day_last_i)].sort_values(by = ['day','point_id'],ascending=[True,True])
        idx_list_i = points_ce_value.index.tolist()
        point_list_i = points_ce_value['point'].tolist()
        day_list_i = points_ce_value['day'].tolist()
        day_unique_list_i = sorted(list(set(day_list_i)))

        ### find common arguments ###
        entity_list_i = []
        for item in point_list_i:
            if type(item) is str:
                class_doc = nlp(item)
                entity_list_i.append([ent.text for ent in class_doc.ents if ent.label_ in ent_type])
            else:
                # For debug
                entity_list_i.append([])
        
        entity_date_dict_i = dict()
        for k in range(len(entity_list_i)):
            for item in entity_list_i[k]:
                if item not in entity_date_dict_i.keys():
                    entity_date_dict_i[item] = []
                entity_date_dict_i[item].append(day_list_i[k])
        
        ### get potential conbinations of points ###
        entity_key_list = list(entity_date_dict_i.keys())
        potential_locs_list = []
        for key_i in entity_key_list:
            date_list_i = sorted(list(set(entity_date_dict_i[key_i])))
            if len(date_list_i)<3:
                potential_locs_list.append([])
                continue
            num_qa = math.floor((len(date_list_i)-1)/2)
            loc_list_triple = []
            for k in range(num_qa):
                three_date_list = date_list_i[k*2: min((k+1)*2+1, len(date_list_i))]
                loc_list_OneDate = []
                for dt in three_date_list:
                    loc_list_tmp = []
                    for j in range(len(day_list_i)):
                        if (day_list_i[j]==dt) and (key_i in entity_list_i[j]):
                            loc_list_tmp.append(j)
                    loc_list_OneDate.append(loc_list_tmp)
                loc_list_triple.extend([[k1, k2, k3] for k1 in loc_list_OneDate[0] for k2 in loc_list_OneDate[1] for k3 in loc_list_OneDate[2]])
            potential_locs_list.append(loc_list_triple)                
                
        ### post verify and save ###
        for k in range(len(entity_key_list)):
            print(f"Common entity: {entity_key_list[k]}, Num of potential comb: {len(potential_locs_list[k])}")
            if potential_locs_list[k]==[]:
                continue
            ### Set the maximun num of potential comb for one common entity to be no more than 20
            num_limit = 10
            for triple in random.sample(potential_locs_list[k], min(len(potential_locs_list[k]), num_limit)):
                print(triple)
                idx_list = [idx_list_i[loc] for loc in triple]
                point_i_str = '\n'.join([f"{ki+1}. {point_list_i[triple[ki]].strip()}" for ki in range(len(triple))])
                post_verify = orderqa_post_verify(point_i_str)
                qual_score = post_verify.compute_quality_bool()
                num_tokens_input.append(40083)
                num_tokens_output.append(15*3)
                print(f"qual_score: {qual_score}")
                if qual_score==0:
                    continue
                points_list_save = [points_ce_value.loc[ki]['point'] for ki in idx_list]
                print(points_list_save)
                day_list_save = [str(points_ce_value.loc[ki]['day']) for ki in idx_list]
                point_ids_list_save = [points_ce_value.loc[ki]['point_id'] for ki in idx_list]
                (choices_unorder_i, ground_truth_i) = random_choices(points_list_save)
                writer.writerow([ce_val_i, entity_key_list[k], point_ids_list_save, '\n'.join(points_list_save), ','.join(day_list_save), choices_unorder_i, ground_truth_i])