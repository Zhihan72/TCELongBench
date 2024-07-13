import os
import csv
import pandas as pd

QA_order = pd.read_csv('./dataset/TLB_order_creation.csv')
ce_id_list = QA_order['ce_id'].unique().tolist()
index_del_list = []
for ce_val_i in ce_id_list:
    idx_list_i = QA_order[QA_order['ce_id']==ce_val_i].index.tolist()
    if len(idx_list_i)==1:
        continue
    points_each_qa = []
    for idx_i in idx_list_i:
        if type(QA_order.loc[idx_i]['choices'])==str:
            points_list_i = [item[2:].strip() for item in QA_order.loc[idx_i]['choices'].split('\n') if len(item.strip())!=0]
            points_each_qa.append(points_list_i)
        else:
            index_del_list.append(idx_i)
            points_each_qa.append([])
    len_list = list(range(len(points_each_qa)))
    pair_list = list(itertools.combinations(len_list, 2))
    for loc_1, loc_2 in pair_list:
        points_1 = points_each_qa[loc_1]
        idx_1 = idx_list_i[loc_1]
        points_2 = points_each_qa[loc_2]
        idx_2 = idx_list_i[loc_2]
        points_common = [item for item in points_1 if item in points_2]
        if len(points_common)>1:
            index_del_list.append(idx_2)

index_del_list_unique = list(set(index_del_list))
QA_order = QA_order.drop(index_del_list_unique)
save_path = './dataset/TLB_order_deduplication.csv'
QA_order.to_csv(save_path, sep=',', index=False, header=True)