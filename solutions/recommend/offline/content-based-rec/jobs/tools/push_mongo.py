import sys
import argparse

if __package__ is None:
    sys.path.append('..')
    from common import push_mongo
else:
    from jobs.common import push_mongo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mongo-uri", type=str, required=True
    )
    parser.add_argument(
        "--mongo-database", type=str, required=True
    )
    parser.add_argument(
        "--mongo-collection", type=str, required=True
    )
    parser.add_argument(
        "--data", type=str, required=True
    )
    parser.add_argument(
        "--fields", type=str, required=True
    )
    parser.add_argument(
        "--index-fields", type=str, default=""
    )
    parser.add_argument(
        "--write-mode", type=str, default="append", choices=["append", "overwrite"]
    )
    args = parser.parse_args()
    return args


def main(args):
    fields = [n for n in args.fields.split(',')]
    index_fields = [n for n in args.index_fields.split(',') if n in fields]

    push_mongo(args.mongo_uri, args.mongo_database, args.mongo_collection, 
        args.data, fields, index_fields, args.write_mode)


if __name__ == '__main__':
    args = parse_args()
    main(args)

