import os
import json
import pickle
import random
import argparse

from schema import User, Item

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

def similar(user_tags, item_tags, weights=[], metric='cosine'):
    user_tags = set(user_tags)
    item_tags = set(item_tags)
    if not user_tags or not item_tags:
        return 0.0
    if metric == 'jaccard':
        return len(user_tags & item_tags) / len(user_tags | item_tags)
    elif metric == 'cosine':
        return len(user_tags & item_tags) / ((len(user_tags)**0.5)*(len(item_tags)**0.5))
    return len(user_tags & item_tags) / len(user_tags | item_tags)  # jaccard as default

def retrieval(user, tag2items, topk):
    weights = user.tag_weights if user.tag_weights and len(user.tag_weights) == len(user.tags) else [1.0 for i in range(len(user.tags))]
    y = sum(weights) + 0.0000001
    weights = [x/y for x in weights]
    item_ids = set()
    for tag, weight in zip(user.tags, weights):
        n = int(topk*weight)
        item_ids.update(tag2items.get(tag, [])[:n])
    return list(item_ids)

def rerank(user, item2tags, cand_item_ids, topk):
    user_tags, user_tag_weights = user.tags, user.tag_weights
    item_scores = []
    for item_id in cand_item_ids:
        score = similar(user_tags, item2tags.get(item_id, []), user_tag_weights)
        item_scores.append([item_id, score])
    return sorted(item_scores, key=lambda x:x[1], reverse=True)[:topk]

def search(users, item2tags, tag2items, topk):
    ret = []
    for user in users:
        if not user.tags:
            # user doesn't have any tags
            ret.append([])
        else:
            item_ids = retrieval(user, tag2items, topk*10)
            item_scores = rerank(user, item2tags, item_ids, topk)
            ret.append(item_scores)
    return ret

def main():
    args = parse_args()

    users = User.load_from_json(args.user_data)

    with open(args.item_data, 'rb') as f:
        item_data = pickle.load(f)
    index, rindex = item_data['index'], item_data['rindex']

    hits = search(users, index, rindex, args.topk)

    res = {}
    for i, user in enumerate(users):
        user_res = {"items": [], "scores": []}
        for item_id, score in hits[i]:
            user_res["items"].append(item_id)
            user_res["scores"].append(score)
        res[user.user_id] = user_res

    with open(args.result_data, 'w', encoding='utf8') as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    main()
