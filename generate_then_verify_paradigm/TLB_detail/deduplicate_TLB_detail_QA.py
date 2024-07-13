import os
import pandas as pd
import csv
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder

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

MCQ_detail = pd.read_csv("./dataset/TLB_detail_creation.csv")
ce_id_list = MCQ_detail['ce_id'].unique().tolist()
index_del_list = []

for ce_val_i in ce_id_list:
    idx_list_i = MCQ_detail[MCQ_detail['ce_id']==ce_val_i].index.tolist()
    question_list_i = MCQ_detail[MCQ_detail['ce_id']==ce_val_i]['question'].tolist()
    for id_j in range(len(question_list_i)):
        idx_j = idx_list_i[id_j]
        score_dup = model_dup.predict([(question_list_i[id_j], item) for item in question_list_i]).tolist()
        dup_loc = [loc for loc in range(len(score_dup)) if score_dup[loc] >= 0.8]
        if len(dup_loc)==1:
            continue
        dup_idx = [idx_list_i[item] for item in dup_loc]
        for item in dup_idx:
            if item > idx_j:
                index_del_list.append(item)

index_del_list_unique = list(set(index_del_list))
MCQ_detail = MCQ_detail.drop(index_del_list_unique)
save_path = './dataset/TLB_detail_deduplication.csv'
MCQ_detail.to_csv(save_path, sep=',', index=False, header=True)