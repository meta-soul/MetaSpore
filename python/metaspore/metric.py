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
import torch
import struct
from datetime import datetime
from ._metaspore import ModelMetricBuffer

class ModelMetric(object):
    def __init__(self):
        self._instance_num = 0

    @property
    def instance_count(self):
        return self._instance_num

    def _get_scalar_pack_info(self):
        return (('_instance_num', 'l'),)

    def _get_array_pack_info(self):
        return ()

    def clear(self):
        self._instance_num = 0

    def merge(self, other):
        self._instance_num += other._instance_num

    def accumulate(self, *, batch_size, **kwargs):
        self._instance_num += batch_size

    def _get_scalar_pack_format(self):
        fmt = ''.join(t for n, t in self._get_scalar_pack_info())
        return fmt

    def _get_scalar_values(self):
        values = tuple(getattr(self, n) for n, t in self._get_scalar_pack_info())
        return values

    def _pack_scalar_values(self):
        fmt = self._get_scalar_pack_format()
        values = self._get_scalar_values()
        data = struct.pack(fmt, *values)
        data = numpy.array(tuple(data), dtype=numpy.uint8)
        return data

    def _unpack_scalar_values(self, data):
        fmt = self._get_scalar_pack_format()
        values = struct.unpack(fmt, data)
        for (name, tag), value in zip(self._get_scalar_pack_info(), values):
            setattr(self, name, value)

    def get_states(self):
        states = self._pack_scalar_values(),
        for name in self._get_array_pack_info():
            states += getattr(self, name),
        return states

    def from_states(self, states):
        self._unpack_scalar_values(states[0])
        for name, array in zip(self._get_array_pack_info(), states[1:]):
            field = getattr(self, name)
            field[...] = array

    def _as_numpy_ndarray(self, value):
        if isinstance(value, numpy.ndarray):
            return value
        elif isinstance(value, torch.Tensor):
            return value.data.numpy()
        else:
            message = f"value must be numpy ndarray or torch tensor; {value!r} is invalid"
            raise TypeError(message)

    def __str__(self):
        string = '#instance={self.instance_count}'
        return string

    def _get_format_header(self):
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return string

    def _get_format_body(self, delta):
        string = f'#instance: {self.instance_count}'
        return string

    def format(self, delta):
        header = self._get_format_header()
        body = self._get_format_body(delta)
        string = header + ' -- ' + body
        return string

class BasicModelMetric(ModelMetric):
    def __init__(self):
        super().__init__()
        self._loss_sum = 0.0

    def _get_scalar_pack_info(self):
        return super()._get_scalar_pack_info() + (('_loss_sum', 'd'),)

    def clear(self):
        super().clear()
        self._loss_sum = 0.0

    def merge(self, other):
        super().merge(other)
        self._loss_sum += other._loss_sum

    def accumulate(self, *, batch_loss, **kwargs):
        super().accumulate(**kwargs)
        self._loss_sum += batch_loss

    def compute_loss(self):
        if self.instance_count == 0:
            return float('nan')
        return self._loss_sum / self.instance_count

    def __str__(self):
        string = f'loss={self.compute_loss()}'
        string += f', {super().__str__()}'
        return string

    def _get_format_body(self, delta):
        string = f'loss: {self.compute_loss()}'
        string += f', \u0394loss: {delta.compute_loss()}'
        string += f', {super()._get_format_body(delta)}'
        return string

class BinaryClassificationModelMetric(BasicModelMetric):
    def __init__(self, buffer_size=1000000, threshold=0.0, beta=1.0):
        super().__init__()
        self._buffer_size = buffer_size
        self._threshold = threshold
        self._beta = beta
        self._positive_buffer = numpy.zeros(buffer_size, dtype=numpy.float64)
        self._negative_buffer = numpy.zeros(buffer_size, dtype=numpy.float64)
        self._prediction_sum = 0.0
        self._label_sum = 0.0
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

    def _get_scalar_pack_info(self):
        return super()._get_scalar_pack_info() + (
            ('_prediction_sum', 'd'),
            ('_label_sum', 'd'),
            ('_true_positive', 'l'),
            ('_true_negative', 'l'),
            ('_false_positive', 'l'),
            ('_false_negative', 'l'))

    def _get_array_pack_info(self):
        return super()._get_array_pack_info() + (
            '_positive_buffer',
            '_negative_buffer')

    def clear(self):
        super().clear()
        self._positive_buffer.fill(0.0)
        self._negative_buffer.fill(0.0)
        self._prediction_sum = 0.0
        self._label_sum = 0.0
        self._true_positive = 0
        self._true_negative = 0
        self._false_positive = 0
        self._false_negative = 0

    def merge(self, other):
        super().merge(other)
        self._positive_buffer += other._positive_buffer
        self._negative_buffer += other._negative_buffer
        self._prediction_sum += other._prediction_sum
        self._label_sum += other._label_sum
        self._true_positive += other._true_positive
        self._true_negative += other._true_negative
        self._false_positive += other._false_positive
        self._false_negative += other._false_negative

    def accumulate(self, *, predictions, labels, **kwargs):
        super().accumulate(**kwargs)
        predictions = self._as_numpy_ndarray(predictions)
        labels = self._as_numpy_ndarray(labels)
        if labels.dtype != numpy.float32:
            labels = labels.astype(numpy.float32)
        ModelMetricBuffer.update_buffer(self._positive_buffer, self._negative_buffer,
                                        predictions, labels)
        self._prediction_sum += predictions.sum()
        self._label_sum += labels.sum()
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
        string += f', {super().__str__()}'
        if self.threshold > 0.0:
            string += f', accuracy={self.compute_accuracy()}'
            string += f', precision={self.compute_precision()}'
            string += f', recall={self.compute_recall()}'
            string += f', F{self.beta:g}_score={self.compute_f_score()}'
        return string

    def _get_format_body(self, delta):
        string = f'auc: {self.compute_auc()}'
        string += f', \u0394auc: {delta.compute_auc()}'
        string += f', pcoc: {self.compute_pcoc()}'
        string += f', \u0394pcoc: {delta.compute_pcoc()}'
        string += f', {super()._get_format_body(delta)}'
        if self.threshold > 0.0:
            string += f', accuracy: {self.compute_accuracy()}'
            string += f', precision: {self.compute_precision()}'
            string += f', recall: {self.compute_recall()}'
            string += f', F{self.beta:g}_score: {self.compute_f_score()}'
        return string
