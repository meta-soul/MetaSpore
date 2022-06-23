import argparse

from pymilvus import connections, utility, Collection

def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, required=True
    )
    parser.add_argument(
        "--port", type=int, required=True
    )
    parser.add_argument(
        "--collection-name", type=str, required=True
    )
    parser.add_argument(
        "--ann-metric-type", type=str, default="IP"
    )
    parser.add_argument(
        "--ann-index-type", type=str, default="IVF_FLAT"
    )
    parser.add_argument(
        "--ann-param-nlist", type=int, default=1024
    )
    # just for search
    parser.add_argument(
        "--ann-param-nprobe", type=int, default=32
    )
    return parser

def get_collection(host, port, collection_name):
    connections.connect(host=host, port=port)
    if not utility.has_collection(collection_name):
        return None
    collection = Collection(collection_name)
    collection.load()  # 执行向量相似性搜索之前将 collection 加载到内存中，以便在内存中执行检索计算
    return collection

def drop_collection(host, port, collection_name):
    connections.connect(host=host, port=port)
    res = input(f"Do you want to drop collection '{collection_name}'(Y/N): ")
    if res.lower() in ["y", "yes"]:
        utility.drop_collection(collection_name)
        print("It's done!")
        return True
    print("Ok, the collection will be keep, bye!")
    return False
