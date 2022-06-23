import random

from milvus.utils import get_base_parser, get_collection


def parse_args():
    parser = get_base_parser()
    parser.add_argument(
        "--index-field", type=str, required=True
    )
    parser.add_argument(
        "--vector", type=str, required=True, help="The embedding vector string split by comma"
    )
    parser.add_argument(
        "--limit", type=int, default=10
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    collection = get_collection(args.host, args.port, args.collection_name)
    assert collection is not None, "Collection {} not exists!".format(args.collection_name)

    search_params = {
        "metric_type": args.ann_metric_type, 
        "params": {"nprobe": args.ann_param_nprobe}
    }

    if ',' in args.vector:
        vec = [float(v) for v in args.vector.split(',')]
    else:
        vec = [random.random() for i in range(int(args.vector))]
    results = collection.search(
        data=[vec],
        anns_field=args.index_field,
        param=search_params, 
        limit=args.limit,
        expr=None,
        consistency_level="Strong"
    )
    return results

if __name__ == '__main__':
    results = main()
    print(results)
