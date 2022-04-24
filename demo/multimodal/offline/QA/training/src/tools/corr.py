import sys
from scipy.stats import pearsonr, spearmanr

s1_file, s2_file, method = sys.argv[1], sys.argv[2], sys.argv[3]
scores1 = [float(l.strip()) for l in open(s1_file, 'r') if l.strip()]
scores2 = [float(l.strip()) for l in open(s2_file, 'r') if l.strip()]
if method == 'pearson':
    print(pearsonr(scores1, scores2))
elif method == 'spearman':
    print(spearmanr(scores1, scores2))
