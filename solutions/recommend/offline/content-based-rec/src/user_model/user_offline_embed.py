import os
import sys
import json
import pickle
import argparse

import numpy as np
from tqdm import tqdm

from schema import User, Action

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
        '--item-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--dump-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--user-data',
        type=str,
        default=''
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
        '--action-agg-func',
        type=str,
        choices=['latest', 'avg', 'decay'],
        default='decay'
    )
    parser.add_argument(
        '--action-max-len',
        type=int,
        default=30
    )
    parser.add_argument(
        '--action-decay-rate',
        type=float,
        default=0.9
    )
    return parser.parse_args()

def load_json_data(json_file, scene_id, action_type, min_value, max_value, sortby):
    user_ids, user_actions = [], []
    actions = Action.load_from_json(json_file)
    for user_id, u_actions in Action.group_by_user_id(actions, scene_id, action_type, min_value, max_value, sortby):
        user_ids.append(user_id)
        user_actions.append([a.item_id for a in u_actions])
    return user_ids, user_actions

def get_user_emb_from_action(user_emb, action_ids, items, agg_func, max_len, decay_rate):
    if max_len > 0:
        action_ids = action_ids[-max_len:]  # trim to the last actions

    action_embs = [items[item_id] for item_id in action_ids if item_id in items]

    if agg_func == 'latest':
        user_emb = action_embs[-1]
    elif agg_func == 'avg':
        user_emb = np.mean(action_embs, axis=0)
    elif agg_func == 'decay':
        if len(action_embs) == 1:
            user_emb = action_embs[0]
        else:
            for action_emb in action_embs:
                user_emb =decay_rate*user_emb + (1.0-decay_rate)*action_emb

    return user_emb

def main():
    args = parse_args()
    # load item data
    items = {}
    emb_dim = None
    with open(args.item_data, 'rb') as f:
        item_data = pickle.load(f)
    for item_id, item_emb in zip(item_data['ids'], item_data['embs']):
        items[item_id] = item_emb
        emb_dim = item_emb.shape[0]
    print(f"Total items: {len(items)}")
    # load user history emb
    users = {}
    if args.user_data:
        with open(args.user_data, 'rb') as f:
            user_data = pickle.load(f)
        for user_id, user_emb in zip(user_data['ids'], user_data['embs']):
            users[user_id] = user_emb
    print(f"User history: {len(users)}")
    # load user action data
    user_ids, user_actions = load_json_data(args.action_data, args.scene_id, args.action_type,  
        args.action_value_min, args.action_value_max, args.action_sortby_key)
    print(f"Total users: {len(user_ids)}")
    # user encode
    user_embs = []
    for user_id, action_ids in tqdm(zip(user_ids, user_actions)):
        if user_id in users and users[user_id].shape[0] == emb_dim:
            user_emb = users[user_id]
        else:
            user_emb = np.zeros(emb_dim, dtype=np.float32)  # must be a float32

        user_emb = get_user_emb_from_action(user_emb, action_ids, items, 
            args.action_agg_func, args.action_max_len, args.action_decay_rate)

        user_embs.append(user_emb)
    print(f"Total embs: {len(user_embs)}")
    # dump data
    os.makedirs(os.path.dirname(args.dump_data), exist_ok=True)
    with open(args.dump_data, 'wb') as f:
        pickle.dump({"ids": user_ids, "embs": user_embs, "_ids": list(range(len(user_ids)))}, f, 
            protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dump to: {args.dump_data}")


if __name__ == '__main__':
    main()
