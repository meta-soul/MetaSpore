import os
import json
import logging
from typing import Union, Tuple, List, Iterable, Dict, Callable


import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer, models, losses, util

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator, LabelAccuracyEvaluator

logger = logging.getLogger(__name__)

from dataset import STSDataset, NLIDataset, QMCDataset


def create_encoder(model_name_or_path='bert-base-chinese', 
        max_seq_len=256, dense_layer_features=-1, device=None, is_pretrained_model=False, pooling='mean'):
    """
    What is an encoder?

    For sentence / text embeddings, we want to map a variable length input text to a fixed sized dense vector. The most basic network architecture we can use is the following: sentence->BERT->pooling->vector.

    A BERT layer and a pooling layer is one final SentenceTransformer model.

    We feed the input sentence or text into a transformer network like BERT. BERT produces contextualized word embeddings for all input tokens in our text. As we want a fixed-sized output representation (vector u), we need a pooling layer. Different pooling options are available, the most basic one is mean-pooling: We simply average all contextualized word embeddings BERT is giving us. This gives us a fixed 768 dimensional output vector independet how long our input text was.

    :param model_name_or_path: The BERT name or path.
    :param max_seq_len: The max length of sequence.
    :param dense_layer_features: The dim of the last dense fully-connected layer.
    :param device: cpu or cuda:0
    :param is_pretrained_model: Does the model is a SBERT pretrained model.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 如果输入模型是预训练的SBERT则直接返回
    if is_pretrained_model:
        logger.info("Load SentenceTransformer from pre-trained checkpoint.")
        encoder = SentenceTransformer(model_name_or_path, device=device)
        encoder.max_seq_length = max_seq_len
        return encoder

    logger.info("Create SentenceTransformer from scratch.")
    # 对 huggingface AutoModel 的封装，输出是一个字典含有各种粒度、层级的向量，如 `token_embeddings`, `cls_token_embeddings`, `all_layer_embeddings` 等
    # 参见：https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
    layers = []
    bert_model = models.Transformer(model_name_or_path, max_seq_length=max_seq_len)
    layers.append(bert_model)

    # pooling 策略将变长序列转换为固定长度表征向量，输出是一个字典，其中 `sentence_embedding` 对应 pooling 后的语句表征
    # 参见：https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
    pooling_mode_mean_tokens = False
    pooling_mode_max_tokens = False
    pooling_mode_cls_token = False
    if pooling == 'max':
        pooling_mode_max_tokens = True
    elif pooling == 'cls':
        pooling_mode_cls_token = True
    else:
        pooling_mode_mean_tokens = True
    pooling_model = models.Pooling(bert_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=pooling_mode_mean_tokens,
                                   pooling_mode_max_tokens=pooling_mode_max_tokens,
                                   pooling_mode_cls_token=pooling_mode_cls_token)
    layers.append(pooling_model)

    # 全连接层对语句表征进行非线性变换，也即仅变换 `sentence_embedding`
    # 参见：https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Dense.py
    if dense_layer_features > 0:
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
            out_features=dense_layer_features, activation_function=nn.Tanh())
        layers.append(dense_model)

    # modules 就是 SentenceTransformer 内部抽取文本语义表征的 pytorch 串形模块，
    # 直接调用 model() 时，等价于串行调用了这些模块。
    encoder = SentenceTransformer(modules=layers, device=device)
    return encoder

def create_evaluator(exp_name, data_file, task_type="sts", batch_size=32, model=None):
    if not data_file or not os.path.isfile(data_file):
        return [], None
    if task_type == "qmc":
        samples = QMCDataset(data_file).load()
        evaluator = BinaryClassificationEvaluator.from_input_examples(samples,
            batch_size=batch_size, name='{}-eval'.format(exp_name))
    elif task_type == "nli":
        samples = NLIDataset(data_file).load()
        dataloader = DataLoader(samples, shuffle=True, batch_size=batch_size)
        evaluator = LabelAccuracyEvaluator(dataloader, softmax_model=model, 
            name='{}-eval'.format(exp_name))
    else:
        # Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation in comparison to the STS's gold standard labels.
        samples = STSDataset(data_file).load()
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, 
            batch_size=batch_size, name='{}-eval'.format(exp_name))
    return samples, evaluator

