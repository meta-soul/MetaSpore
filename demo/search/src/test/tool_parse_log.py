import re
import ast
import sys
import json

epoch_regex = re.compile(r'Epoch: (.*)')
step_regex = re.compile(r'Steps: (.*)')
precision_regex = re.compile(r'Precision: (.*)')
recall_regex = re.compile(r'Recall: (.*)')
mrr_regex = re.compile(r'MRR: (.*)')

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    epoch_s, step_s, precision_s, recall_s, mrr_s = line.split('\t')
    #print(epoch_s, step_s, precision_s, recall_s, mrr_s)
    epoch = epoch_regex.search(epoch_s).groups(0)[0]
    step = step_regex.search(step_s).groups(0)[0]
    precision = ast.literal_eval(precision_regex.search(precision_s).groups(0)[0])
    recall = ast.literal_eval(recall_regex.search(recall_s).groups(0)[0])
    mrr = ast.literal_eval(mrr_regex.search(mrr_s).groups(0)[0])
    #print(epoch, step, precision, recall, mrr)
    precision_list = [precision[f"precision@{k}"] for k in [1, 10, 50]]
    recall_list = [recall[f"recall@{k}"] for k in [1, 10, 50]]
    mrr_list = [mrr[f"mrr@{k}"] for k in [1, 10, 50]]
    precision_list = [f"{v:.2%}" for v in precision_list]
    recall_list = [f"{v:.2%}" for v in recall_list]
    mrr_list = [f"{v:.2%}" for v in mrr_list]
    print(epoch, step, *precision_list, *recall_list, *mrr_list, sep='\t')

