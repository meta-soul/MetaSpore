import argparse

import pymongo


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri", type=str, required=True
    )
    #parser.add_argument(
    #    "--host", type=str, required=True
    #)
    #parser.add_argument(
    #    "--port", type=int, required=True
    #)
    parser.add_argument(
        "--db-name", type=str, required=True
    )
    parser.add_argument(
        "--collection-name", type=str, required=True
    )
    return parser

def create_mongo_session(host="127.0.0.1", port=27017, uri=""):
    if uri:
        return pymongo.MongoClient(uri)
    return pymongo.MongoClient(host, port)
