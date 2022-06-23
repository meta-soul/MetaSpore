import os
import sys
import pickle
import argparse

from tqdm import tqdm
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

from milvus.utils import get_base_parser, get_collection

def create_schema_from_data(item, desc=''):
    fields = []
    for k, v in item.items():
        if k in ['id', 'item_id']:
            fields.append(FieldSchema(
                name=k, 
                dtype=DataType.INT64, 
                is_primary=True, 
            ))
        elif k.endswith('_emb'):
            fields.append(FieldSchema(
                name=k,
                dtype=DataType.FLOAT_VECTOR,
                dim=len(v)
            ))
        elif isinstance(v, int):
            fields.append(FieldSchema(
                name=k,
                dtype=DataType.INT64
            ))
        elif isinstance(v, float):
            fields.append(FieldSchema(
                name=k,
                dtype=DataType.FLOAT
            ))
    schema = CollectionSchema(fields=fields, description=desc)
    return schema

def insert_into_collection(collection, data_iter, fields, batch_size=1024):
    data = [[] for i in range(len(fields))]
    for item in tqdm(data_iter):
        for i in range(len(fields)):
            data[i].append(item[fields[i]])
        if len(data[0]) == batch_size:
            collection.insert(data)
            data = [[] for i in range(len(fields))]
    if not data and not data[0]:
        collection.insert(data)


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
        "--index-field", type=str, default="item_emb"
    )
    parser.add_argument(
        "--collection-desc", type=str, default=""
    )
    parser.add_argument(
        "--collection-shards", type=int, default=2
    )
    args = parser.parse_args()
    return args

def load_index_data(index_file, id_key, emb_key, idx_key):
    with open(index_file, 'rb') as fin:
        data = pickle.load(fin)
    for item_id, item_emb in zip(data[id_key], data[emb_key]):
        item = {}
        item['id'] = item_id
        item[idx_key] = item_emb
        yield item

def main(args):
    print("Loading data...")
    data_iter = iter(load_index_data(args.index_file, args.id_field, args.emb_field, args.index_field))

    item = next(data_iter)

    #print(utility.list_collections())
    print("\nConnect milvus connection...")
    collection = get_collection(args.host, args.port, args.collection_name)

    if collection is None:
        print("\nCreating milvus collection...")
        schema = create_schema_from_data(item, desc=args.collection_desc)
        collection = Collection(
            name=args.collection_name,
            schema=schema, 
            shards_num=args.collection_shards
        )
        print("\nCreated collection: {}".format(collection.schema))

    schema = collection.schema.to_dict()
    fields = [x['name'] for x in schema['fields']]
    index_field = args.index_field

    print("\nInsert into collection...")
    # insert a single data
    data = []
    for i in range(len(fields)):
        values = []
        values.append(item[fields[i]])
        data.append(values)
    collection.insert(data)
    # insert batch data 
    insert_into_collection(collection, data_iter, fields)

    print("\nBuilding index...")
    index_params = {
        "metric_type": args.ann_metric_type,
        "index_type": args.ann_index_type,
        "params": {"nlist": args.ann_param_nlist}
    }
    ret = collection.create_index(field_name=index_field, index_params=index_params)
    print(f"\nResults: {ret}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
