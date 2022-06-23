import os
import sys
import math
import json
import pickle
import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm

from schema import User, Action, Item

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
        '--action-max-len',
        type=int,
        default=30
    )
    parser.add_argument(
        '--user-tags-topk',
        type=int,
        default=10
    )
    return parser.parse_args()

def update_user_by_action(user, actions, max_len, topk):
    if max_len > 0:
        actions = actions[-max_len:]
    cnt = Counter()
    if user.tag_weights and len(user.tag_weights) == len(user.tags):
        for t, n in zip(user.tags, user.tag_weights):
            cnt[t] += math.ceil(n)
    else:
        cnt.update(user.tags)
    for action in actions:
        if action.item is not None and len(action.item.tags) > 0:
            cnt.update(action.item.tags)
    user.tags = [x[0] for x in cnt.most_common(topk)]
    user.tag_weights = [x[1] for x in cnt.most_common(topk)]
    return user

def main():
    args = parse_args()
    # load action data
    actions = Action.load_from_json(args.action_data)
    print(f"Total actions: {len(actions)}")
    # load item data
    items = Item.load_from_json(args.item_data)
    print(f"Total items: {len(items)}")
    # load user history emb
    users = {}
    if args.user_data:
        for user in User.load_from_json(args.user_data):
            users[user.user_id] = user
    print(f"User history: {len(users)}")
    # join user&item
    actions = Action.join(actions, users=users, items=items)
    # group by user
    group_users = Action.group_by_user_id(actions, args.scene_id, args.action_type,
        args.action_value_min, args.action_value_max, args.action_sortby_key)
    # user encode
    user_list = []
    for user_id, user_actions in group_users:
        if user_id not in users:
            users[user_id] = User(user_id)
        user = update_user_by_action(users[user_id], actions, args.action_max_len, 
            args.user_tags_topk)
        user_list.append(user)
    print(f"User dump: {len(user_list)}")
    # dump data
    os.makedirs(os.path.dirname(args.dump_data), exist_ok=True)
    User.save_to_json(user_list, args.dump_data)
    print(f"Dump to: {args.dump_data}")


if __name__ == '__main__':
    main()
