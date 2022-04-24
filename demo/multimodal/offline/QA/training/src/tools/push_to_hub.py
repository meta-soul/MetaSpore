import os
import sys
import argparse

from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="The path of model.")
parser.add_argument("--repo", required=True, help="The repo name for your model hub")
parser.add_argument("--org", default="DMetaSoul", help="Organization in which you want to push your model")
parser.add_argument("--msg", default="commit", help="Message to commit while pushing")
parser.add_argument("--force", action="store_true", help="wether saving to an existing repository is OK")
args = parser.parse_args()

model = SentenceTransformer(args.model)
hub_url = model.save_to_hub(args.repo, organization=args.org, commit_message=args.msg, exist_ok=args.force)
print(f"Hub URL: {hub_url}")
