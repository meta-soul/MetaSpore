import argparse
import logging

import torch
import faiss
import numpy as np
from tqdm import tqdm

from data import create_cross_encoder_dataloader
from modeling import TransformerDualEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help='The pretrained model name or path.'
    )
    parser.add_argument(
        '--input-file',
        required=True,
        type=str,
        help='The input query/passage text file, with fields (query, title, paragraph, label).'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        type=str,
        help='The embedding output file.'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default='faiss:FlatIP',
        choices=['numpy', 'faiss:FlatIP', 'faiss:FlatL2', 'faiss:IVFFlat', 'faiss:IVFPQ'],
        help='The embedding output will be saved as numpy or faiss index file.'
    )
    parser.add_argument(
        '--text-indices',
        type=str,
        default='0,1',
        help='The text columns index of the input file, split by comma.'
    )
    parser.add_argument(
        '--text-max-len',
        type=int,
        default=256,
        help='The max sequence length of text.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=-1,
        help='The max batches for inference.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4
    )
    args = parser.parse_args()
    return args

def faiss_index(embs, emb_dim, index_mode):
    if index_mode == 'FlatL2':
        index = faiss.IndexFlatL2(emb_dim)
        index.add(embs)
        #return index.ntotal
    else:
        #elif index_mode == 'FlatIP':
        index = faiss.IndexFlatIP(emb_dim)
        index.add(embs)
        #return index.ntotal
    return index

def main():
    args = parse_args()

    logging.info("Args:")
    for name in vars(args):
        value = getattr(args, name)
        logging.info(f"\t{name}={value}")
        #print(f"\t{name}={value}")

    torch.set_num_threads(args.num_workers)

    # load model
    logging.info("Loading model")
    model = TransformerDualEncoder.load_pretrained(args.model,
        device=args.device, max_seq_len=args.text_max_len)

    # load data
    logging.info("Loading data")
    dataloader = create_cross_encoder_dataloader(args.input_file, model.tokenize, 
        text_indices=[int(i) for i in args.text_indices.split(',')],
        batch_size=args.batch_size, device=args.device, 
        num_workers=0, shuffle=False)  # must be shuffle=False for inference embeddings

    # inference
    logging.info("Starting inference")
    text_embs = []
    n = 0
    with torch.no_grad():
        for features, labels in tqdm(dataloader):
            outputs = model(**features)
            embs = outputs['sentence_embedding']
            embs = embs.cpu().numpy() if embs.is_cuda else embs.numpy()
            text_embs.append(embs.astype('float32'))
            n += 1
            if args.num_batches > 0 and n >= args.num_batches:
                break
    text_embs = np.concatenate(text_embs)

    # dump
    logging.info("Dumping embedding")
    b, h = text_embs.shape
    if args.output_format.startswith('faiss'):
        index_mode = 'FlatIP'
        if ':' in args.output_format:
            index_mode = args.output_format.split(':')[1]
        output_file = args.output_file if args.output_file.endswith('.faiss') else args.output_file + '.faiss'
        index = faiss_index(text_embs, h, index_mode)
        faiss.write_index(index, output_file)
    else:
        output_file = args.output_file if args.output_file.endswith('.npy') else args.output_file + '.npy'
        np.save(output_file, text_embs)


if __name__ == '__main__':
    main()
