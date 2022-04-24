import scipy
import numpy as np

class STSDataset(object):
    
    def __init__(self, data_file):
        self._data_file = data_file

    def load(self):
        texts1, texts2, labels = [], [], []
        with open(self._data_file, 'r', encoding='utf8') as fin:
            for line in fin:
                line = line.strip('\r\n')
                if not line:
                    continue
                text1, text2, label = line.split('\t')
                label = float(label)
                texts1.append(text1)
                texts2.append(text2)
                labels.append(label)
        return texts1, texts2, labels

def compute_kernel_bias(vecs, n_components=256):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    引用自：https://kexue.fm/archives/8069
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    引用自：https://kexue.fm/archives/8069
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def compute_corrcoef(x, y):
    return scipy.stats.spearmanr(x, y).correlation
