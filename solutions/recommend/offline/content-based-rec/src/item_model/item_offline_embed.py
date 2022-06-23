import os
import sys
import json
import pickle
import argparse

from sentence_transformers import SentenceTransformer

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
    parser.add_argument(
        '--text-model-name',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )
    parser.add_argument(
        '--shard-size',
        type=int,
        default=0
    )
    return parser.parse_args()

def load_json_data(json_file):
    item_ids, item_texts = [], []
    for item in Item.load_from_json(json_file):
        item_ids.append(item.item_id)
        item_texts.append(item.content)
    return item_ids, item_texts

def main():
    args = parse_args()
    # load model
    model = SentenceTransformer(args.text_model_name, device=args.device)
    # load raw data
    item_ids, item_texts = load_json_data(args.item_data)
    # encode item
    item_embs = model.encode(item_texts, batch_size=args.batch_size,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    # dump data
    os.makedirs(os.path.dirname(args.dump_data), exist_ok=True)
    with open(args.dump_data, 'wb') as f:
        pickle.dump({"ids": item_ids, "embs": item_embs, "_ids": list(range(len(item_ids)))}, f, 
            protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
