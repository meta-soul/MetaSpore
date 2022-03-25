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

import numpy
import struct
from ._metaspore import ModelMetricBuffer

class ModelMetric(object):
    def __init__(self, buffer_size=1000000, threshold=0.0, beta=1.0):
        self._buffer_size = buffer_size
        self._threshold = threshold
        self._beta = beta
        self._positive_buffer = numpy.zeros(buffer_size, dtype=numpy.float64)
        self._negative_buffer = numpy.zeros(buffer_size, dtype=numpy.float64)
        self._prediction_sum = 0.0
        self._label_sum = 0.0
        self._instance_num = 0
        self._true_positive = 0
        self._true_negative = 0
        self._false_positive = 0
        self._false_negative = 0

    @property
    def threshold(self):
        return self._threshold

    @property
    def beta(self):
        return self._beta

    @property
    def instance_count(self):
        return self._instance_num

    def clear(self):
        self._positive_buffer.fill(0.0)
        self._negative_buffer.fill(0.0)
        self._prediction_sum = 0.0
        self._label_sum = 0.0
        self._instance_num = 0
        self._true_positive = 0
        self._true_negative = 0
        self._false_positive = 0
        self._false_negative = 0

    def merge(self, other):
        self._positive_buffer += other._positive_buffer
        self._negative_buffer += other._negative_buffer
        self._prediction_sum += other._prediction_sum
        self._label_sum += other._label_sum
        self._instance_num += other._instance_num
        self._true_positive += other._true_positive
        self._true_negative += other._true_negative
        self._false_positive += other._false_positive
        self._false_negative += other._false_negative

    def accumulate(self, predictions, labels):
        if labels.dtype != numpy.float32:
            labels = labels.astype(numpy.float32)
        ModelMetricBuffer.update_buffer(self._positive_buffer, self._negative_buffer,
                                        predictions, labels)
        self._prediction_sum += predictions.sum()
        self._label_sum += labels.sum()
        self._instance_num += len(labels)
        if self.threshold > 0.0:
            predicted_positive = predictions > self._threshold
            predicted_negative = predictions <= self._threshold
            actually_positive = labels > self._threshold
            actually_negative = labels <= self._threshold
            self._true_positive += (predicted_positive & actually_positive).sum()
            self._true_negative += (predicted_negative & actually_negative).sum()
            self._false_positive += (predicted_positive & actually_negative).sum()
            self._false_negative += (predicted_negative & actually_positive).sum()

    def compute_auc(self):
        auc = ModelMetricBuffer.compute_auc(self._positive_buffer, self._negative_buffer)
        return auc

    def compute_pcoc(self):
        if self._label_sum == 0.0:
            return float('nan')
        return self._prediction_sum / self._label_sum

    def compute_accuracy(self):
        num1 = self._true_positive + self._true_negative
        num2 = self._false_positive + self._false_negative
        num = num1 + num2
        if num == 0:
            return float('nan')
        return num1 / num

    def compute_precision(self):
        num = self._true_positive + self._false_positive
        if num == 0:
            return float('nan')
        return self._true_positive / num

    def compute_recall(self):
        num = self._true_positive + self._false_negative
        if num == 0:
            return float('nan')
        return self._true_positive / num

    def compute_f_score(self):
        precision = self.compute_precision()
        recall = self.compute_recall()
        num1 = precision * recall
        num = (self._beta ** 2) * precision + recall
        if num == 0.0:
            return float('nan')
        f_score = (1 + self._beta ** 2) * num1 / num
        return f_score

    def __str__(self):
        string = f'auc={self.compute_auc()}'
        string += f', pcoc={self.compute_pcoc()}'
        if self.threshold > 0.0:
            string += f', accuracy={self.compute_accuracy()}'
            string += f', precision={self.compute_precision()}'
            string += f', recall={self.compute_recall()}'
            string += f', F{self.beta:g}_score={self.compute_f_score()}'
        return string

    def _get_pack_format(self):
        return 'ddl' + 'l' * 4

    def get_states(self):
        scalars = self._prediction_sum,
        scalars += self._label_sum,
        scalars += self._instance_num,
        scalars += self._true_positive,
        scalars += self._true_negative,
        scalars += self._false_positive,
        scalars += self._false_negative,
        scalars = struct.pack(self._get_pack_format(), *scalars)
        scalars = numpy.array(tuple(scalars), dtype=numpy.uint8)
        states = scalars,
        states += self._positive_buffer,
        states += self._negative_buffer,
        return states

    @classmethod
    def from_states(cls, states):
        scalars, pos_buf, neg_buf = states
        buffer_size = len(pos_buf)
        inst = cls(buffer_size)
        inst._positive_buffer[:] = pos_buf
        inst._negative_buffer[:] = neg_buf
        pred_sum, lab_sum, inst_num, tp, tn, fp, fn = struct.unpack(inst._get_pack_format(), scalars)
        inst._prediction_sum = pred_sum
        inst._label_sum = lab_sum
        inst._instance_num = inst_num
        inst._true_positive = tp
        inst._true_negative = tn
        inst._false_positive = fp
        inst._false_negative = fn
        return inst
