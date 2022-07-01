import sys
import argparse

if __package__ is None:
    sys.path.append('..')
    from common import push_milvus
else:
    from jobs.common import push_milvus


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--milvus-host", type=str, required=True
    )
    parser.add_argument(
        "--milvus-port", type=int, required=True
    )
    parser.add_argument(
        "--milvus-collection", type=str, required=True
    )
    parser.add_argument(
        "--data", type=str, required=True
    )
    parser.add_argument(
        "--fields", type=str, required=True
    )
    parser.add_argument(
        "--id-field", type=str, default="id"
    )
    parser.add_argument(
        "--emb-field", type=str, default="emb"
    )
    parser.add_argument(
        "--collection-desc", type=str, default=""
    )
    parser.add_argument(
        "--collection-shards", type=int, default=2
    )
    parser.add_argument(
        "--write-batch", type=int, default=1024
    )
    parser.add_argument(
        "--write-interval", type=float, default=0.1
    )
    parser.add_argument(
        "--index-type", type=str, default="IVF_FLAT"
    )
    parser.add_argument(
        "--index-metric", type=str, default="IP"
    )
    parser.add_argument(
        "--index-nlist", type=int, default=1024
    )
    args = parser.parse_args()
    return args


def main(args):
    fields = args.fields.split(',')
    push_milvus(args.milvus_host, args.milvus_port, args.milvus_collection, args.data, fields, 
        args.id_field, args.emb_field, args.collection_desc, 
        args.collection_shards, args.write_batch, args.write_interval, 
        args.index_type, args.index_metric, args.index_nlist)


if __name__ == '__main__':
    args = parse_args()
    main(args)
