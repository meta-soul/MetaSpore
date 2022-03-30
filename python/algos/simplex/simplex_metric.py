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
import metaspore as ms
from metaspore.metric import ModelMetric
from metaspore._metaspore import ModelMetricBuffer

class SimpleXModelMetric(ms.ModelMetric):
    def __init__(self, buffer_size=1000000, threshold=0.0, beta=1.0):
        super().__init__(buffer_size=1000000, threshold=0.0, beta=1.0)
        self._loss = 0

    def clear(self):
        super().clear()
        self._loss = 0

    def merge(self, other):
        super().merge(other)
        self._loss += other._loss

    def accumulate(self, predictions, labels, loss):
        if labels.dtype != numpy.float32:
            labels = labels.astype(numpy.float32)
        # ModelMetricBuffer.update_buffer(self._positive_buffer, self._negative_buffer,
        #                                 predictions, labels)
        self._prediction_sum += predictions.sum()
        self._label_sum += labels.sum()
        self._instance_num += len(labels)
        self._loss += loss.sum() * len(labels)
        if self.threshold > 0.0:
            predicted_positive = predictions > self._threshold
            predicted_negative = predictions <= self._threshold
            actually_positive = labels > self._threshold
            actually_negative = labels <= self._threshold
            self._true_positive += (predicted_positive & actually_positive).sum()
            self._true_negative += (predicted_negative & actually_negative).sum()
            self._false_positive += (predicted_positive & actually_negative).sum()
            self._false_negative += (predicted_negative & actually_positive).sum()
        
    def compute_loss(self):
        if self._instance_num==0:
            return float('nan')
        return self._loss / self._instance_num

    def _get_pack_format(self):
        return 'ddl' + 'l' * 4 + 'd'

    def get_states(self):
        scalars = self._prediction_sum,
        scalars += self._label_sum,
        scalars += self._instance_num,
        scalars += self._true_positive,
        scalars += self._true_negative,
        scalars += self._false_positive,
        scalars += self._false_negative,
        scalars += self._loss,
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
        pred_sum, lab_sum, inst_num, tp, tn, fp, fn, loss = struct.unpack(inst._get_pack_format(), scalars)
        inst._prediction_sum = pred_sum
        inst._label_sum = lab_sum
        inst._instance_num = inst_num
        inst._true_positive = tp
        inst._true_negative = tn
        inst._false_positive = fp
        inst._false_negative = fn
        inst._loss = loss
        return inst
