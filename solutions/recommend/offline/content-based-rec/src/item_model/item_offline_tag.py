import os
import sys
import json
import pickle
import argparse

from schema import Item

def parse_args():
    parser = argparse.ArgumentParser()
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
    #parser.add_argument(
    #    '--shard-size',
    #    type=int,
    #    default=0
    #)
    return parser.parse_args()

def load_json_data(json_file):
    item_ids, item_tags, item_weights = [], [], []
    for item in Item.load_from_json(json_file):
        item_ids.append(item.item_id)
        item_tags.append(item.tags)
        item_weights.append(item.weight)
    return item_ids, item_tags, item_weights

def main():
    args = parse_args()
    # load raw data
    item_ids, item_tags, item_weights = load_json_data(args.item_data)
    # reverse index
    weights = {}
    index, rindex = {}, {}
    for item_id, tags, weight in zip(item_ids, item_tags, item_weights):
        index[item_id] = tags
        weights[item_id] = weight
        for tag in tags:
            if tag not in rindex:
                rindex[tag] = set()
            rindex[tag].add(item_id)
    # rindex sort by item weight
    for tag in rindex:
        item_ids = list(rindex[tag])
        item_ids = sorted(item_ids, key=lambda x:weights[x], reverse=True)
        rindex[tag] = item_ids
    # dump data
    os.makedirs(os.path.dirname(args.dump_data), exist_ok=True)
    with open(args.dump_data, 'wb') as f:
        pickle.dump({"index": index, "rindex": rindex}, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
