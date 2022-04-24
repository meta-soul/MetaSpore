import os
import re
import random
import logging

import tqdm
from sentence_transformers import InputExample
from sentence_transformers import datasets as sentence_transformers_datasets
import numpy as np
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer


def default_label_converter(label):
    if '.' in label:
        return float(label)
    else:
        return int(label) 

class DataAugmentFunc(object):

    # Deletion noise.
    @staticmethod
    def delete(text, del_ratio=0.6):
        if re.search(u'[\u4e00-\u9fff]', text):
            words = list(text)  # for chinese
        else:
            words = nltk.word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True # guarantee that at least one word remains
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed

    @staticmethod
    def repetition(text, dup_ratio=0.32):
        if re.search(u'[\u4e00-\u9fff]', text):
            sep = ''
            words = list(text)  # for chinese
        else:
            sep = ' '
            #words = nltk.word_tokenize(text)
            words = text.split()
        n = len(words)
        if n == 0:
            return text

        dup_len = random.randint(0, max([2, int(n*dup_ratio)]))
        i_list = list(range(0, n))
        random.shuffle(i_list)
        dup_word_index = i_list[:dup_len]

        rep_words = []
        for i, word in enumerate(words):
            rep_words.append(word)
            if i in dup_word_index:
                rep_words.append(word)
        
        return sep.join(rep_words)

def make_pos_neg_pairs(sentences, pos_neg_ratio=8):
    random.shuffle(sentences)
    samples = []
    sentence_idx = 0
    while sentence_idx + 1 < len(sentences):
        s1 = sentences[sentence_idx]
        if len(samples) % pos_neg_ratio > 0:    #Negative (different) pair
            sentence_idx += 1
            s2 = sentences[sentence_idx]
            label = 0
        else:   #Positive (identical pair)
            s2 = sentences[sentence_idx]
            label = 1

        sentence_idx += 1
        samples.append([s1, s2, label])

    return samples

#################################
# pair形式的监督数据集
#################################
class PairDataset(object):

    def __init__(self, data_file):
        self.data_file = data_file

    def process(self, row):
        return row

    def postprocess(self, *args):
        if len(args) == 0:
            return None
        elif len(args) == 1:
            return args[0]
        else:
            return args

    def load(self, max_samples=-1, label_converter=default_label_converter, with_label=True, split_label=False):
        data = []
        num_fields = 3 if with_label else 2
        with open(self.data_file, 'r', encoding='utf8') as fin:
            for line in tqdm.tqdm(fin):
                line = line.strip('\r\n')
                if not line:
                    continue
                fields = line.split('\t')
                if len(fields) < num_fields:
                    continue
                row = []
                if not with_label:
                    query, doc = fields[:num_fields]
                    row = [query, doc]
                else:
                    query, doc, label = fields[:num_fields]
                    label = label_converter(label)
                    row = [query, doc, label]
                row = self.process(row)
                if row:
                    data.append(row)
                if max_samples > 0 and len(data) > max_samples:
                    break

        if not with_label or not split_label:
            return self.postprocess(data)

        sent_pairs = [[x[0], x[1]] for x in data]
        labels = [x[2] for x in data]
        return self.postprocess(sent_pairs, labels)


class PairWithClsLabelDataset(PairDataset):
    """分类任务的pair数据集"""

    def __init__(self, data_file, label2id={}):
        super(PairWithClsLabelDataset, self).__init__(data_file)
        self.label2id = label2id

    def load(self, max_samples=-1, split_label=False, label_converter=int):
        if self.label2id:
            label_converter = lambda x:self.label2id[x]
        data = super(PairWithClsLabelDataset, self).load(max_samples=max_samples, 
            label_converter=label_converter, with_label=True,  split_label=split_label) 
        random.shuffle(data)
        if split_label:
            return data
        else:
            return [InputExample(texts=[x[0], x[1]], label=x[2]) for x in data]

class PairWithRegLabelDataset(PairDataset):
    """回归任务的pair数据集"""

    def __init__(self, data_file):
        super(PairWithRegLabelDataset, self).__init__(data_file)

    def load(self, max_samples=-1, split_label=False, label_converter=float):
        data = super(PairWithRegLabelDataset, self).load(max_samples=max_samples, 
            label_converter=label_converter, with_label=True, split_label=split_label) 
        random.shuffle(data)
        if split_label:
            return data
        else:
            return [InputExample(texts=[x[0], x[1]], label=x[2]) for x in data]

class SSMDataset(PairWithClsLabelDataset):
    """对称的文本对匹配数据集，label=0|1"""

    def process(self, row):
        label = row[2]
        return None if label not in [0, 1] else row

    def load(self, max_samples=-1, losses={}):
        examples = super(SSMDataset, self).load(max_samples=max_samples, split_label=False)
        if not losses:
            return examples
        datasets = []
        if 'contrastive' in losses:
            datasets.append(examples)
        if 'cosine' in losses:
            datasets.append([InputExample(texts=x.texts, label=float(x.label)) for x in examples])
        if 'circle' in losses:
            datasets.append(examples)
        if 'logistic' in losses:
            datasets.append([InputExample(texts=x.texts, label=float(x.label)) for x in examples])
        if 'ranking' in losses:
            dataset = []
            for x in examples:
                if x.label == 0:
                    continue
                s1, s2 = x.texts
                dataset.append(InputExample(texts=[s1, s2], label=1))
                dataset.append(InputExample(texts=[s2, s1], label=1))
            random.shuffle(dataset)
            datasets.append(dataset)
        return datasets

class ASMDataset(PairWithClsLabelDataset):
    """非对称的文本对匹配数据集，label=0|1"""

    def process(self, row):
        label = row[2]
        return None if label not in [0, 1] else row

    def load(self, max_samples=-1, losses={}):
        examples = super(ASMDataset, self).load(max_samples=max_samples, split_label=False)
        if not losses:
            return examples
        datasets = []
        if 'contrastive' in losses:
            datasets.append(examples)
        if 'cosine' in losses:
            datasets.append([InputExample(texts=x.texts, label=float(x.label)) for x in examples])
        if 'circle' in losses:
            datasets.append(examples)
        if 'logistic' in losses:
            datasets.append(examples)
        if 'ranking' in losses:
            dataset = []
            for x in examples:
                if x.label == 0:
                    continue
                s1, s2 = x.texts
                dataset.append(InputExample(texts=[s1, s2], label=1))
            random.shuffle(dataset)
            datasets.append(dataset)
        return datasets


class QMCDataset(SSMDataset):
    """问题对匹配数据集（二分类）"""
    pass


class QDRDataset(ASMDataset):
    """query和doc相关性数据集（二分类）"""
    pass


class NLIDataset(PairWithClsLabelDataset):
    """自然语言推理（NLI）数据集（三分类）"""

    def process(self, row):
        label = row[2]
        return None if label not in [0, 1, 2] else row

    def load(self, max_samples=-1, losses={}):
        examples = super(NLIDataset, self).load(max_samples=max_samples, split_label=False)
        if not losses:
            return examples
        datasets = []
        if 'softmax3' in losses:
            datasets.append(examples)
        if 'circle' in losses:
            datasets.append(examples)
        if 'ranking' in losses:
            premises = {}
            for x in examples:
                if x.label == 1:
                    continue
                p, h = x.texts
                if p not in premises:
                    premises[p] = {'pos': '', 'neg': ''}
                if x.label == 2:
                    premises[p]['pos'] = h
                else:
                    premises[p]['neg'] = h
            dataset = []
            for p in premises:
                pos, neg = premises[p]['pos'], premises[p]['neg']
                if (not pos) or (not neg):
                    continue
                dataset.append(InputExample(texts=[p, pos, neg]))
            random.shuffle(dataset)
            datasets.append(dataset)
        return datasets


class STSDataset(PairWithRegLabelDataset):
    """语义相似对（STS）数据集（0.0～1.0）"""

    def process(self, row):
        #label = row[2]/5.0  # scale the score between 0.0 and 1.0
        label = row[2]
        return None if label<0.0 or label>1.0 else row[:2]+[label]

    def load(self, max_samples=-1, losses={}):
        examples = super(STSDataset, self).load(max_samples=max_samples, split_label=False)
        if not losses:
            return examples
        datasets = []
        if 'cosine' in losses:
            datasets.append(examples)
        if 'circle' in losses:
            datasets.append(examples)
        return datasets


############################
# triplet形式的监督数据集
############################
class TripletDataset(object):

    def __init__(self, data_file):
        self.data_file = data_file

    def load(self, max_samples=-1, group_by_query=True, with_logits=False, non_neg=False):
        data = []
        num_fields = 3
        if with_logits:
            num_fields = 5
        if non_neg:
            num_fields = 2
        with open(self.data_file, 'r', encoding='utf8') as fin:
            for line in tqdm.tqdm(fin):
                line = line.strip('\r\n')
                if not line:
                    continue
                fields = line.split('\t')
                if len(fields) < num_fields:
                    continue
                if non_neg:
                    query, pos = fields[:num_fields]
                    data.append([query, pos])
                elif not with_logits:
                    query, pos, neg = fields[:num_fields]
                    data.append([query, pos, neg])
                else:
                    query, pos, neg, pos_score, neg_score = fields[:num_fields]
                    data.append([query, pos, neg, float(pos_score), float(neg_score)])
                if max_samples > 0 and len(data) > max_samples:
                    break

        if not group_by_query:
            return data

        queries = {}
        for row in data:
            query = row[0]
            if query not in queries:
                queries[query] = {'query':query, 'positive': set(), 'negative': set()}
            pos, neg = None, None
            if len(row) >= 3:
                pos, neg = row[1], row[2]
            elif len(row) >= 2:
                pos = row[1]
            if pos:
                queries[query]['positive'].add(pos)
            if neg:
                queries[query]['negative'].add(neg)
        return queries

class TripletWithDistLabelDataset(TripletDataset):
    """(query, positive, negative, pos_score, neg_score)"""

    def load(self, max_samples=-1, losses={}):
        data = super(TripletWithDistLabelDataset, self).load(max_samples=max_samples, group_by_query=False, with_logits=True)
        examples = [InputExample(texts=[x[0], x[1], x[2]], label=x[3]-x[4]) for x in data]
        if not losses:
            examples
        datasets = []
        if 'mmse' in losses:
            datasets.append(examples)
        return datasets


########################
# 无label的纯文本数据集
########################
class CorpusDataset(object):

    def __init__(self, data_file):
        self.data_file = data_file

    def load(self, max_samples=-1, cols=[0], shuffle=True):
        data = []
        with open(self.data_file, 'r', encoding='utf8') as fin:
            for line in tqdm.tqdm(fin):
                line = line.strip('\r\n')
                if not line:
                    continue
                fields = line.split('\t')
                if len(cols) == 1:
                    data.append(fields[cols[0]])
                else:
                    data.append([fields[i] for i in cols])
        if shuffle:
            random.shuffle(data)
        return data

class QuerySingleDataset(CorpusDataset):
    """(query)"""

    def load(self, max_samples=-1, i=0, losses={}, shuffle=True):
        sents = super(QuerySingleDataset, self).load(max_samples=max_samples, cols=[i], shuffle=shuffle)
        if not losses:
            return sents

        datasets = []
        if 'simcse' in losses:
            # copy the same sentence as positive pair
            datasets.append([InputExample(texts=[s, s], label=1) for s in sents])
        if 'esimcse' in losses:
            datasets.append([InputExample(texts=[DataAugmentFunc.repetition(s), s], label=1) for s in sents])
        if 'tsdae' in losses:
            # add deletion noise to make positive pair
            datasets.append([InputExample(texts=[DataAugmentFunc.delete(s), s], label=1) for s in sents])
        if 'ct' in losses:
            # create neg via global sampling
            datasets.append([InputExample(texts=[s1, s2], label=label) for s1,s2,label in make_pos_neg_pairs(sents)])
        if 'ct2' in losses:
            # copy the same sentence as positive pair
            datasets.append([InputExample(texts=[s, s], label=1) for s in sents])
        return datasets


class QueryPositivePairDataset(CorpusDataset):
    """(query, positive)"""

    def load(self, max_samples=-1, i=0, j=1, losses={}, shuffle=True, default_label=1):
        data = super(QueryPositivePairDataset, self).load(max_samples=max_samples, cols=[i, j], shuffle=shuffle)
        if not losses:
            return data

        datasets = []
        if 'ranking' in losses:
            datasets.append([InputExample(texts=[query, pos], label=default_label) for query,pos in data])
        return datasets

class QueryPositiveNegativeTripletDataset(CorpusDataset):
    """(query, positive, negative)"""

    def load(self, max_samples=-1, i=0, j=1, k=2, losses={}, shuffle=True):
        data = super(QueryPositiveNegativeTripletDataset, self).load(max_samples=max_samples, cols=[i, j, k], shuffle=shuffle)
        if not losses:
            return data

        datasets = []
        if 'ranking' in losses:
            datasets.append([InputExample(texts=[query, pos, neg]) for query,pos,neg in data])
        if 'triplet' in losses:
            datasets.append([InputExample(texts=[query, pos, neg]) for query,pos,neg in data])
        return datasets
