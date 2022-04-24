import argparse
import numpy as np
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--model", default="bert-base-chinese")
parser.add_argument("--n-components", default=256, type=int)
parser.add_argument("--eval-file", required=True)
args = parser.parse_args()

layers = []
bert_model = models.Transformer(args.model, max_seq_length=512)
layers.append(bert_model)

# pooling 策略将变长序列转换为固定长度表征向量，输出是一个字典，其中 `sentence_embedding` 对应 pooling 后的语句表征
# 参见：https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
pooling_model = models.Pooling(bert_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_max_tokens=False,
                               pooling_mode_cls_token=False)
layers.append(pooling_model)
encoder = SentenceTransformer(modules=layers)

word_emb_dim = bert_model.get_word_embedding_dimension()

texts1, texts2, labels = STSDataset(args.eval_file).load()
embs1 = encoder.encode(texts1, convert_to_numpy=True)
embs2 = encoder.encode(texts2, convert_to_numpy=True)

kernel, bias = compute_kernel_bias(np.vstack([embs1, embs2]))
if args.n_components > 0 and args.n_components < word_emb_dim:
    kernel = kernel[:, :args.n_components]

embs1 = transform_and_normalize(embs1, kernel, bias)
embs2 = transform_and_normalize(embs2, kernel, bias)
scores = (embs1 * embs2).sum(axis=1)
corrcoef = compute_corrcoef(labels, scores)
print(args.name, corrcoef, sep='\t')
