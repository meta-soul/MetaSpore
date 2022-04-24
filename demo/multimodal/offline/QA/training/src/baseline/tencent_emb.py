import sys
import jieba
import numpy as np
from gensim.models import KeyedVectors

text_file= sys.argv[1] # single or pair
emb_file = sys.argv[2]

n = -1
texts1 = []
texts2 = []
with open(text_file, 'r', encoding='utf8') as fin:
    for line in fin:
        line = line.strip('\r\n')
        if not line:
            continue
        fields = line.split('\t')
        if n < 0:
            n = 1 if len(fields) == 1 else 2
        if n == 1:
            texts1.append(list(jieba.cut(fields[0], cut_all=True)))
        elif n == 2:
            texts1.append(list(jieba.cut(fields[0], cut_all=True)))
            texts2.append(list(jieba.cut(fields[1], cut_all=True)))

wv_from_text = KeyedVectors.load_word2vec_format(emb_file, binary=False)
if n == 1:
    for text in texts1:
        embs = [wv_from_text.word_vec(w) for w in text]
        text_emb = np.vstack(embs).mean(axis=0).tolist()
        print(*text_emb, sep=' ')
elif n == 2:
    for text1, text2 in zip(texts1, texts2):
        dist = wv_from_text.wmdistance(text1, text2)
        print(1-dist)
