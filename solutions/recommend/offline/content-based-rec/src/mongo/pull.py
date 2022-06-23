from mongo.utils import get_base_parser, create_mongo_session

def parse_args():
    parser = get_base_parser()
    parser.add_argument(
        "--object-id", type=str, default=""
    )
    return parser.parse_args()

def main(args):
    client = create_mongo_session(uri=args.uri)
    db = client[args.db_name]
    collection = db[args.collection_name]

    if args.object_id:
        print(collection.find_one({'_id': args.object_id}))
    else:
        print(collection.find_one())

if __name__ == '__main__':
    args = parse_args()
    main(args)
