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

import torch
import metaspore as ms
import torch.nn.functional as F
import json
from datetime import datetime
import struct
import numpy as np
import pandas as pd
from pyspark.sql.functions import col

from metaspore._metaspore import Message

class MMoEAgent(ms.PyTorchAgent):
    '''
    def nansum(self, x):
        return torch.where(torch.isnan(x), torch.zeros_like(x), x).sum(dim=0)

    def log_loss(self, yhat, y):
        return self.nansum(-(y * (yhat + 1e-12).log() + (1 - y) * (1 - yhat + 1e-12).log()))

    def compute_loss(self, predictions, labels):
        loss =  self.log_loss(predictions, labels) / labels.shape[0]
        return loss.sum()
    '''

    def train_minibatch(self, minibatch):
        self.model.train()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions = self.model(ndarrays)
        labels = torch.from_numpy(labels).reshape(-1, len(self.input_label_column_indexes))
        loss = self.compute_loss(predictions, labels)
        self.trainer.train(loss)
        self.update_progress(predictions[:, 0], labels[:, 0], loss)

    def validate_minibatch(self, minibatch):
        self.model.eval()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions = self.model(ndarrays)
        labels = torch.from_numpy(labels).reshape(-1, len(self.input_label_column_indexes))
        self.update_progress(predictions[:, 0], labels[:, 0], torch.tensor(0.0))
        return predictions[:, self.input_label_column_indexes].detach()

    def preprocess_minibatch(self, minibatch):
        import numpy as np
        ndarrays = [col.values for col in minibatch]
        labels_list = []
        for input_label_column_index in self.input_label_column_indexes:
            labels_list.append(np.reshape(minibatch[input_label_column_index].values.astype(np.float32), (-1, 1)))
        labels = np.concatenate(labels_list, axis=1)
        return ndarrays, labels

    def feed_validation_dataset(self):
        df = self.dataset.withColumn('prediction_result', self.feed_validation_minibatch()(*self.dataset.columns))
        for i in range(len(self.input_label_column_indexes)):
            df = df.withColumn(f'rawPrediction_{i}', col(f'prediction_result.col{i}').cast(self.output_prediction_column_type))
            df = df.withColumn(f'label_{i}', df[self.input_label_column_indexes[i]].cast(self.output_label_column_type))
        df = df.drop('prediction_result')
        self.validation_result = df
        df.cache()
        df.write.format('noop').mode('overwrite').save()

    def feed_validation_minibatch(self):
        from pyspark.sql.functions import pandas_udf
        returnTypeStr = ', '.join([f'col{i}: float' for i in range(len(self.input_label_column_indexes))])
        @pandas_udf(returnType=returnTypeStr)
        def _feed_validation_minibatch(*minibatch):
            self = __class__.get_instance()
            result = self.validate_minibatch(minibatch)
            result = self.process_validation_minibatch_result(minibatch, result)
            return result
        return _feed_validation_minibatch

    def process_validation_minibatch_result(self, minibatch, result):
        minibatch_size = len(minibatch[self.input_label_column_index])
        result = pd.DataFrame({
            f'col{i}' : self._to_pd_series(result[:, i], minibatch_size) for i in range(len(self.input_label_column_indexes))
        })
        return result

    def _to_pd_series(self, result, minibatch_size):
        if result is None:
            result = pd.Series([0.0] * minibatch_size)
        if len(result) != minibatch_size:
            message = "result length (%d) and " % len(result)
            message += "minibatch size (%d) mismatch" % minibatch_size
            raise RuntimeError(message)
        if not isinstance(result, pd.Series):
            if len(result.reshape(-1)) == minibatch_size:
                result = result.reshape(-1)
            else:
                message = "result can not be converted to pandas series; "
                message += "result.shape: {}, ".format(result.shape)
                message += "minibatch_size: {}".format(minibatch_size)
                raise RuntimeError(message)
            result = pd.Series(result)
        return result


    def _create_metric(self):
        metric = MMoEMetric()
        return metric

    def update_metric(self, predictions, labels, loss):
        self._metric.accumulate(predictions.data.numpy(), labels.data.numpy(), loss.data.numpy())

    def update_progress(self, predictions, labels, loss):
        self.minibatch_id += 1
        self.update_metric(predictions, labels, loss)
        if self.minibatch_id % self.metric_update_interval == 0:
            self.push_metric()

    def handle_request(self, req):
        import json
        body = json.loads(req.body)
        command = body.get('command')
        if command == 'PushMetric':
            states = ()
            for i in range(req.slice_count):
                states += req.get_slice(i),
            accum = self._metric
            delta = MMoEMetric.from_states(states)
            accum.merge(delta)
            from datetime import datetime
            string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            string += f' -- auc: {accum.compute_auc()}'
            string += f', \u0394auc: {delta.compute_auc()}'
            string += f', loss: {accum.compute_loss()}'
            string += f', \u0394loss: {delta.compute_loss()}'
            string += f', pcoc: {accum.compute_pcoc()}'
            string += f', \u0394pcoc: {delta.compute_pcoc()}'
            string += f', #instance: {accum.instance_count}'
            if accum.threshold > 0.0:
                string += f', accuracy: {accum.compute_accuracy()}'
                string += f', precision: {accum.compute_precision()}'
                string += f', recall: {accum.compute_recall()}'
                string += f', F{accum.beta:g}_score: {accum.compute_f_score()}'
            print(string)
            from metaspore._metaspore import Message
            res = Message()
            self.send_response(req, res)
            return
        super().handle_request(req)


class MMoEMetric(ms.ModelMetric):
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
        super().accumulate(predictions, labels)
        self._loss += loss.sum() * len(labels)

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
        scalars = np.array(tuple(scalars), dtype=np.uint8)
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
