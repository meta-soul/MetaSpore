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

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scikitplot as skplt

# call scipy.stats.ks_2samp to perform two-sample Kolmogorov-Smirnov test.
def ks_2samp(label, prediction):
    data1 = prediction[label==1]
    data2 = prediction[label!=1]
    return scipy.stats.ks_2samp(data1, data2)

# call skplt.metrics.plot_ks_statistic to plot ks curve.
def ks_curve(label, prediction):
    y_probas = np.zeros((len(prediction), 2))
    y_probas[:, 0] = 1 - prediction
    y_probas[:, 1] = prediction
    return skplt.metrics.plot_ks_statistic(y_true=label, y_probas=y_probas)