import argparse

from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--model", default="bert-base-chinese")
parser.add_argument("--pooling", default="mean", choices=["cls", "max", "mean"])
parser.add_argument("--eval-file", required=True)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

layers = []
bert_model = models.Transformer(args.model, max_seq_length=512)
layers.append(bert_model)

# pooling 策略将变长序列转换为固定长度表征向量，输出是一个字典，其中 `sentence_embedding` 对应 pooling 后的语句表征
# 参见：https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
pooling_model = models.Pooling(bert_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=(args.pooling=="mean"),
                               pooling_mode_max_tokens=(args.pooling=="max"),
                               pooling_mode_cls_token=(args.pooling=="cls"))
layers.append(pooling_model)
encoder = SentenceTransformer(modules=layers, device=args.device)


texts1, texts2, labels = STSDataset(args.eval_file).load()
evaluator_sts = EmbeddingSimilarityEvaluator(texts1, texts2, labels)
score = evaluator_sts(encoder)
print(args.name, score, sep='\t')
