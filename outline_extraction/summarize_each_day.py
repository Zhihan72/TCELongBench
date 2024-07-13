import os
import json
import csv
import pandas as pd
from random import sample
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensor_parallel import TensorParallelPreTrainedModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = './xgen-7b-8k-inst'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
header = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
)

f = open('./doc_filtered.json', 'r')
content = f.read()
GDELT_doc = json.loads(content)
f.close()

f = open('./TCE_ce_id.txt', 'r')
content = f.read()
ce_id_list = [int(item) for item in content.split(',')]
f.close()

doc_summary = pd.read_csv("./doc_summary.csv")

save_path = './day_summary.csv'
with open(save_path, 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['ce_id', 'day', 'Summary', 'Md5_list'])
    for ce_id in ce_id_sample:
        print(f"Writing {ce_id}...")
        dc_ce_id_sample = dc[dc['ce_id']==ce_id]
        Md5_list_ce_id_sample = dc_ce_id_sample['Md5_list'].unique().tolist()
        info_ce_id_sample = pd.DataFrame(data=None,columns=['Md5', 'day', 'Title', 'Text', 'Summary',])
        flag = 0
        for id_ in range(len(dc_ce_id_sample)):
            Md5_list = dc_ce_id_sample.iloc[id_]['Md5_list'].split(', ')
            for Md5_id in Md5_list:
                if Md5_id in list(info_ce_id_sample['Md5'].unique()):
                    continue
                else:
                    title = GDELT_doc[Md5_id]['Title']
                    text = '\n '.join(GDELT_doc[Md5_id]['Text'])
                    day = dc_ce_id_sample.iloc[id_]['day']
                    summary = doc_summary[doc_summary['Md5']==Md5_id]['summary'].tolist()[0]
                    info_ce_id_sample.loc[flag] = [Md5_id, day, title, text, summary,]
                    flag += 1
        day_list = info_ce_id_sample['day'].unique().tolist()
        for day in day_list:
            summary_list = info_ce_id_sample[info_ce_id_sample['day']==day]['Summary'].tolist()
            if len(summary_list) == 1:
                writer.writerow([ce_id, day, summary_list[0], ','.join(info_ce_id_sample[info_ce_id_sample['day']==day]['Md5'].tolist())])
            else:
                summary_cont = '\n\n '.join(summary_list)
                prompt = f"""### Human: 
                Below are the articles in one day. They may have some overlaps with each other. Please summarize all of them in one paragraph with the following rules: 
                1. You should keep the information closely related to the common information across these articles.
                2. The information you summarize should be events that have already happened, instead of those that may happen in the future.
                3. You are not allowed to use any pronoun in the summary. You should replace any pronoun with the specific name of the entity it refers to.
                4. The sentences in your summary should be independent and temporally ordered.
                \n{summary_cont}.\n###
                """
                input_ = header + prompt
                inputs = tokenizer(input_, return_tensors="pt").to(device)
                sample = model.generate(**inputs, do_sample=True, max_new_tokens=2048, top_k=100, eos_token_id=50256)
                summary_ = tokenizer.decode(sample[0]).strip().split('Assistant:')[-1].replace("<|endoftext|>", "").replace("!", "")
                writer.writerow([ce_id, day, summary_, ','.join(info_ce_id_sample[info_ce_id_sample['day']==day]['Md5'].tolist())])