import os
import math
import json
import argparse

from schema import Action

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--scene-id',
        type=str,
        required=True
    )
    parser.add_argument(
        '--action-type',
        type=str,
        required=True
    )
    parser.add_argument(
        '--predict-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--result-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--action-value-min',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--action-value-max',
        type=float,
        default=float('inf')
    )
    parser.add_argument(
        '--action-sortby-key',
        type=str,
        choices=['action_time', 'action_value'],
        default='action_time'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=20
    )
    return parser.parse_args()

def load_pred_data(file):
    with open(file, 'r', encoding='utf8') as f:
        preds = json.load(f)
    return preds

def load_action_data(json_file, scene_id, action_type, min_value, max_value, sortby):
    labels = {}
    actions = Action.load_from_json(json_file)
    for user_id, u_actions in Action.group_by_user_id(actions, scene_id, action_type, min_value, max_value, sortby):
        item_ids = [a.item_id for a in u_actions]
        labels[user_id] = {"items": item_ids}
    return labels

def metric(pred_ids, label_ids, rels=None, topk=20):
    p = 0.0
    if len(pred_ids) > 0:
        p = len(set(pred_ids[:topk]) & set(label_ids)) / topk
    r = 0.0
    if len(label_ids) > 0:
        r = len(set(pred_ids[:topk]) & set(label_ids)) / len(label_ids)
    ap = 0.0
    if len(pred_ids) > 0 and len(label_ids) > 0:
        label_set = set(label_ids)
        n = min(len(pred_ids), topk)
        i = 0
        x, y = 0, 0.0
        while i < n:
            if pred_ids[i] in label_set:
                x += 1
                y += x / (i+1)
            i += 1
        ap = y / min(len(label_ids), topk)
    ndcg = 0.0
    if len(label_ids) > 0:
        binary_rel = rels is None
        rel_map = {} if rels is None else {i:r for i,r in zip(label_ids, rels)}
        label_set = set(label_ids)
        n = min(max(len(label_set), len(pred_ids)), topk)
        dcg, max_dcg = 0.0, 0.0
        i = 0
        while i < n:
            if binary_rel:
                gain = 1.0 / math.log(i + 2)
                if i < len(pred_ids) and pred_ids[i] in label_set:
                    dcg += gain
                if i < len(label_set):
                    max_dcg += gain
            else:
                if i < len(pred_ids):
                    dcg += (math.pow(2.0, rel_map.get(pred_ids[i], 0.0)) - 1) / math.log(i + 2)
                if i < len(label_set):
                    max_dcg += (math.pow(2.0, rel_map.get(label_ids[i], 0.0)) - 1) / math.log(i + 2)
            i += 1
        ndcg = 0.0 if max_dcg == 0.0 else dcg/max_dcg
    return p, r, ap, ndcg

def evaluate(all_preds, all_labels, topk):
    mean_p, mean_r, mean_ap, mean_ndcg = 0.0, 0.0, 0.0, 0.0
    n = 0
    for preds, labels in zip(all_preds, all_labels):
        p, r, ap, ndcg = metric(preds, labels, topk=topk)
        n += 1
        mean_p += p
        mean_r += r
        mean_ap += ap
        mean_ndcg += ndcg
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return mean_p/n, mean_r/n, mean_ap/n, mean_ndcg/n

def main():
    args = parse_args()

    preds = load_pred_data(args.predict_data)
    labels = load_action_data(args.action_data, args.scene_id, args.action_type, 
        args.action_value_min, args.action_value_max, args.action_sortby_key)
    print(f"Pred users: {len(preds)}")
    print(f"True users: {len(labels)}")

    user_ids = set(preds.keys()) & set(labels.keys())
    all_preds = [preds[uid]['items'] for uid in user_ids]
    all_labels = [labels[uid]['items'] for uid in user_ids]
    print(f"Eval users: {len(user_ids)}")

    p, r, ap, ndcg = evaluate(all_preds, all_labels, args.topk)
    res = {
        "precision": p,
        "recall": r,
        "map": ap,
        "ndcg": ndcg
    }
    with open(args.result_data, 'w', encoding='utf8') as f:
        print(json.dumps(res, ensure_ascii=False, indent=4), file=f)
    print(f"Eval result: {res}")

if __name__ == '__main__':
    main()

    #data = [
    #    ([1, 6, 2, 7, 8, 3, 9, 10, 4, 5], [1, 2, 3, 4, 5]),
    #    ([4, 1, 5, 6, 2, 7, 3, 8, 9, 10], [1, 2, 3]),
    #    ([1, 2, 3, 4, 5], [])
    #]
    #print(evaluate([x[0] for x in data], [x[1] for x in data], 2))
