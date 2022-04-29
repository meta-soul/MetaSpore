#
# Copyright 2022 DMetaSoul
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from PIL import Image

from modeling import CLIPImageEncoder
from indexing.base import Builder, get_builder_args_parser


class ImageBuilder(Builder):

    def encode(self, encoder, docs, batch_size=32, device='cpu', **kwargs):
        batch = []
        embeddings = []
        encoder.eval()
        with torch.no_grad():
            for item in docs:
                batch.append(Image.open(item).convert('RGB'))
                if len(batch) == batch_size:
                    embs = encoder.encode(batch, batch_size=batch_size, device=device)['image_embedding']
                    embeddings.append(embs)
                    # close image
                    _ = [img.close() for img in batch]
                    batch = []
            if batch:
                embs = encoder.encode(batch, batch_size=batch_size, device=device)['image_embedding']
                embeddings.append(embs)
        return torch.cat(embeddings).cpu()

def parse_args():
    parser = get_builder_args_parser()
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    torch.set_num_threads(args.num_threads)

    shard_size = args.shard_size
    batch_size = args.batch_size
    index_key, emb_key = args.doc_key_index.split(':')
    values_key = args.doc_key_values.split(',')
    
    encoder = CLIPImageEncoder(args.model)
    encoder.eval()
    encoder.to(args.device)
    
    builder = ImageBuilder(index_key, emb_key, values_key, shard_size)

    doc_iter = builder.load(args.doc_file)

    builder.build(encoder, doc_iter, args.index_file, encode_kwargs={'batch_size': args.batch_size, 'device': args.device})

if __name__ == '__main__':
    main()
