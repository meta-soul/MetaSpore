import sys
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

def load_emb(emb_file):
    embs = []
    with open(emb_file, 'r') as fin:
        for line in fin:
            line = line.strip('\r\n').strip()
            if not line:
                continue
            values = [float(v) for v in line.split(' ')]
            embs.append(values)
    return embs

emb1_file, emb2_file = sys.argv[1], sys.argv[2]
embs1 = load_emb(emb1_file)
embs2 = load_emb(emb2_file)
scores = cosine_similarity(embs1, embs2)
for i in range(len(embs1)):
    print(scores[i][i])
