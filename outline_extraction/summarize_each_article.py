import os
import json
import csv
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
GDELT_filter_doc = json.loads(content)
f.close()

save_summary_file = './doc_summary.csv'
with open(save_summary_file, 'w', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['Md5','title', 'article', 'summary'])
    for Md5_value in list(GDELT_filter_doc.keys()):
        article = '\n '.join(GDELT_filter_doc[Md5_value]['Text'])
        prompt = f"### Human: Please summarize the article with the following rules: \n 1.The summary should be one paragraph.\n 2. The summary should include the key characters, their behaviours or actions and the key timestamps.\n\n{article}.\n###"
        inputs = tokenizer(header + prompt, return_tensors="pt").to(device)
        sample = model.generate(**inputs, do_sample=True, max_new_tokens=2048, top_k=100, eos_token_id=50256)
        output = tokenizer.decode(sample[0]).strip().replace("Assistant:", "")
        summary = output.replace("<|endoftext|>", "").split('### ')[-1]
        writer.writerow([Md5_value, GDELT_filter_doc[Md5_value]['Title'], article, summary])