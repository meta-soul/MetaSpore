import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np

from data import create_cross_encoder_dataloader
from modeling import TransformerDualEncoder

if __name__ == '__main__':
    data_file = sys.argv[1]

    device = 'cuda:0'
    model = TransformerDualEncoder.load_pretrained('DMetaSoul/sbert-chinese-general-v2')
    model.to(device)

    dataloader = create_cross_encoder_dataloader(data_file, 
        model.tokenize, text_indices=[0], device=device, batch_size=16, shuffle=False)

    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(**features)
            embs = outputs['sentence_embedding']
            embs = embs.cpu().numpy() if embs.is_cuda else embs.numpy()
            print(embs)
            break

