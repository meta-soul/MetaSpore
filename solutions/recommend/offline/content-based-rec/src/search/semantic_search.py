import os
import json
import pickle
import argparse

import numpy as np
from sentence_transformers.util import semantic_search, cos_sim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--user-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--item-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--result-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=30
    )
    return parser.parse_args()

def search(user_embs, item_embs, topk, query_chunk_size=100, corpus_chunk_size=100000):
    if isinstance(user_embs, list):
        user_embs = np.stack(user_embs)
    if isinstance(item_embs, list):
        item_embs = np.stack(item_embs)
    hits = semantic_search(user_embs, item_embs, 
        query_chunk_size=query_chunk_size, corpus_chunk_size=corpus_chunk_size,
        top_k=topk, score_function=cos_sim)
    ret = []
    for res_list in hits:
        ret.append([[res['corpus_id'], res['score']] for res in res_list])
    return ret

def main():
    args = parse_args()

    with open(args.user_data, 'rb') as f:
        user_data = pickle.load(f)
    user_ids, user_embs = user_data['ids'], user_data['embs']

    with open(args.item_data, 'rb') as f:
        item_data = pickle.load(f)
    item_ids, item_embs = item_data['ids'], item_data['embs']

    hits = search(user_embs, item_embs, args.topk)

    res = {}
    for i, user_id in enumerate(user_ids):
        user_res = {"items": [], "scores": []}
        for j, score in hits[i]:
            item_id = item_ids[j]
            user_res["items"].append(item_id)
            user_res["scores"].append(score)
        res[user_id] = user_res

    with open(args.result_data, 'w', encoding='utf8') as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    main()
