from indexing.milvus.utils import get_base_parser, get_collection, drop_collection

def parse_args():
    parser = get_base_parser()
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    drop_collection(args.host, args.port, args.collection_name)

if __name__ == '__main__':
    results = main()
