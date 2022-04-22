import torch

from modeling import TextTransformerEncoder
from indexing.base import Builder, get_builder_args_parser


class TextBuilder(Builder):

    def encode(self, encoder, docs, batch_size=32, **kwargs):
        batch = []
        embeddings = []
        for text in docs:
            batch.append(text)
            if len(batch) == batch_size:
                embs = encoder.encode(batch)['sentence_embedding']
                embeddings.append(embs)
                batch = []
        if batch:
            embs = encoder.encode(batch)['sentence_embedding']
            embeddings.append(embs)
        return torch.cat(embeddings)

def parse_args():
    parser = get_builder_args_parser()
    parser.add_argument(
        "--max-seq-len", type=int, default=256
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    torch.set_num_threads(args.num_threads)

    shard_size = args.shard_size
    batch_size = args.batch_size
    index_key, emb_key = args.doc_key_index.split(':')
    values_key = args.doc_key_values.split(',')
    
    encoder = TextTransformerEncoder(args.model, device=args.device)
    
    builder = TextBuilder(index_key, emb_key, values_key, shard_size)

    doc_iter = builder.load(args.doc_file)

    builder.build(encoder, doc_iter, args.index_file, encode_kwargs={'batch_size': args.batch_size})

if __name__ == '__main__':
    main()
