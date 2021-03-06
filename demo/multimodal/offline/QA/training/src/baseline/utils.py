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
    """compute kernel and bias
    vecs.shape = [num_samples, embedding_size]，
    y = (x + bias).dot(kernel)
    
    adopted from：https://kexue.fm/archives/8069
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    """transform and normalize
    
    adopted from：https://kexue.fm/archives/8069
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def compute_corrcoef(x, y):
    return scipy.stats.spearmanr(x, y).correlation
