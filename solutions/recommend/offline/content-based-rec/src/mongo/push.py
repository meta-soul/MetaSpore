import pickle

import pymongo
from tqdm import tqdm

from mongo.utils import get_base_parser, create_mongo_session


def parse_args():
    parser = get_base_parser()
    parser.add_argument(
        "--index-file", type=str, required=True
    )
    parser.add_argument(
        "--id-field", type=str, required=True
    )
    parser.add_argument(
        "--emb-field", type=str, required=True
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048
    )
    args = parser.parse_args()
    return args

def write_into_mongo(collection, items, id_field):
    result = collection.insert_many(items)
    collection.create_index([(id_field, pymongo.ASCENDING)], unique=True)
    return result.inserted_ids

def batchify(data, batch_size):
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def load_index_data(index_file, id_key, emb_key):
    with open(index_file, 'rb') as fin:
        data = pickle.load(fin)
    keys = data.keys()
    for i in range(len(data[id_key])):
        item = {}
        for key in keys:
            if key == emb_key:
                value = data[key][i].tolist()
            else:
                value = data[key][i]
            item[key] = value
        yield item

def main(args):
    print("Loading data...")
    data_iter = iter(load_index_data(args.index_file, args.id_field, args.emb_field))

    print("Mongo collection...")
    client = create_mongo_session(uri=args.uri)
    db = client[args.db_name]
    collection = db[args.collection_name]

    items = [item for item in data_iter]
    print(f"Writing into MongoDB total {len(items)} records...")
    # write by batch to avoid OOM
    for batch in tqdm(batchify(items, args.batch_size)):
        write_into_mongo(collection, batch, args.id_field)

if __name__ == '__main__':
    args = parse_args()
    main(args)

