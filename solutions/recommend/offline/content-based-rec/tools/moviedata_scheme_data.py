import os
import re
import csv
import json
import argparse

from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rating-data',
        type=str,
        default='data/moviedata-10m-split/test.csv'
    )
    parser.add_argument(
        '--movies-data',
        type=str,
        default='data/moviedata-10m-split/movie.json'
    )
    parser.add_argument(
        '--user-data',
        type=str,
        default='data/moviedata-10m-schema/user.test.json'
    )
    parser.add_argument(
        '--item-data',
        type=str,
        default='data/moviedata-10m-schema/item.test.json'
    )
    parser.add_argument(
        '--action-data',
        type=str,
        default='data/moviedata-10m-schema/action.test.json'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # actions
    users = {}
    actions = []
    item_ids = set()
    with open(args.rating_data, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            item_ids.add(row['movie_id'])
            actions.append({
                'scene_id': "douban-cb-100",
                'user_id': row['user_id'],
                'item_id': row['movie_id'],
                'action_type': 'rating',
                'action_time': int(row.get('timestamp', 0)),
                'action_value': row['rating']
            })
            #user_tags = [tag.lower() for tag in row['last_genre'].split('\x01')]
            if row['user_id'] not in users:
                users[row['user_id']] = {'tags': Counter()}
            #users[row['user_id']]['tags'].update(user_tags)

    # items
    items = []
    item2tags = {}
    with open(args.movies_data, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip('\r\n')
            if not line:
                continue
            item = json.loads(line)
            item_id = item['movie_id']
            if item_id not in item_ids:
                continue
            tags = list(set(item['genres']+item['tags']))
            items.append({
                'item_id': item_id,
                'title': item['title'],
                'content': item['title'] + ". " + ", ".join(item['tags'])  + ". " + item['description'],
                'tags': tags
            })
            item2tags[item_id] = tags

    # user tags
    tags_topk = 10
    for action in actions:
        user_id = action['user_id']
        item_id = action['item_id']
        if user_id not in users or item_id not in item2tags:
            continue
        users[user_id]['tags'].update(item2tags[item_id])
    for user_id in users:
        tags = [x[0] for x in users[user_id]['tags'].most_common(tags_topk)]
        weights = [x[1] for x in users[user_id]['tags'].most_common(tags_topk)]
        users[user_id]['tags'] = tags
        users[user_id]['tag_weights'] = weights

    os.makedirs(os.path.dirname(args.user_data), exist_ok=True)

    with open(args.user_data, 'w', encoding='utf8') as f:
        for user_id, user in users.items():
            print(json.dumps({'user_id': user_id, 'tags': user['tags'], 'tag_weights': user['tag_weights']}, ensure_ascii=False), file=f)

    with open(args.item_data, 'w', encoding='utf8') as f:
        for item in items:
            print(json.dumps(item, ensure_ascii=False), file=f)

    with open(args.action_data, 'w', encoding='utf8') as f:
        for action in actions:
            print(json.dumps(action, ensure_ascii=False), file=f)


if __name__ == '__main__':
    main()
