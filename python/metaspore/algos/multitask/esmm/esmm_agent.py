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

import metaspore as ms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import struct

from pyspark.sql.functions import col

class ESMMAgent(ms.PyTorchAgent):
    def __init__(self,
                 ctr_loss_weight=1.0,
                 ctcvr_loss_weight=1.0,
                 **kwargs):
        super().__init__()
        self.ctr_loss = nn.BCELoss()
        self.ctcvr_loss = nn.BCELoss()
        self.ctr_loss_weight = ctr_loss_weight
        self.ctcvr_loss_weight = ctcvr_loss_weight

    def train_minibatch(self, minibatch):
        self.model.train()
        ndarrays, ctcvr_labels, ctr_labels = self.preprocess_minibatch(minibatch)
        ctr_predictions, cvr_predictions = self.model(ndarrays)
        ctcvr_predictions = ctr_predictions * cvr_predictions
        ctcvr_labels = torch.from_numpy(ctcvr_labels).reshape(-1, 1)
        ctr_labels = torch.from_numpy(ctr_labels).reshape(-1, 1)
        loss = self.ctcvr_loss(ctcvr_predictions, ctcvr_labels) * self.ctcvr_loss_weight \
               + self.ctr_loss(ctr_predictions, ctr_labels) * self.ctr_loss_weight
        self.trainer.train(loss)
        self.update_progress(ctcvr_predictions, ctcvr_labels, loss)

    def validate_minibatch(self, minibatch):
        self.model.eval()
        ndarrays, ctcvr_labels, ctr_labels = self.preprocess_minibatch(minibatch)
        ctr_predictions, cvr_predictions = self.model(ndarrays)
        ctcvr_predictions = cvr_predictions * ctr_predictions
        ctcvr_labels = torch.from_numpy(ctcvr_labels).reshape(-1, 1)
        ctr_labels = torch.from_numpy(ctr_labels).reshape(-1, 1)
        self.update_progress(ctcvr_predictions, ctcvr_labels, torch.tensor(0.0))
        return ctcvr_predictions.detach().reshape(-1), \
               ctr_predictions.detach().reshape(-1), \
               cvr_predictions.detach().reshape(-1)


    def preprocess_minibatch(self, minibatch):
        ndarrays = [col.values for col in minibatch]
        ctcvr_labels = minibatch[self.input_label_column_index].values.astype(np.float32)
        ctr_labels = minibatch[self.input_ctr_label_column_index].values.astype(np.float32)
        return ndarrays, ctcvr_labels, ctr_labels

    def feed_validation_dataset(self):
        df = self.dataset.withColumn('prediction_result', self.feed_validation_minibatch()(*self.dataset.columns))\
                         .withColumn(self.output_prediction_column_name, col('prediction_result.col0')) \
                         .withColumn(self.output_ctr_prediction_column_name, col('prediction_result.col1')) \
                         .withColumn(self.output_cvr_prediction_column_name, col('prediction_result.col2'))
        # ctcvr prediction result
        df = df.withColumn(self.output_label_column_name,
                           df[self.input_label_column_index].cast(self.output_label_column_type))
        df = df.withColumn(self.output_prediction_column_name,
                           df[self.output_prediction_column_name].cast(self.output_prediction_column_type))
        # ctr prediction result
        df = df.withColumn(self.output_ctr_label_column_name,
                           df[self.input_ctr_label_column_index].cast(self.output_label_column_type))
        df = df.withColumn(self.output_ctr_prediction_column_name,
                           df[self.output_ctr_prediction_column_name].cast(self.output_prediction_column_type))
        # cvr prediction result
        df = df.withColumn(self.output_cvr_label_column_name,
                           df[self.input_cvr_label_column_index].cast(self.output_label_column_type))
        df = df.withColumn(self.output_cvr_prediction_column_name,
                           df[self.output_cvr_prediction_column_name].cast(self.output_prediction_column_type))
        # validation result
        self.validation_result = df
        # PySpark DataFrame & RDD is lazily evaluated.
        # We must call ``cache`` here otherwise PySpark will try to reevaluate
        # ``validation_result`` when we use it, which is not possible as the
        # PS system has been shutdown.
        df.cache()
        df.write.format('noop').mode('overwrite').save()

    def feed_validation_minibatch(self):
        from pyspark.sql.functions import pandas_udf
        @pandas_udf(returnType='col0: float, col1: float, col2: float')
        def _feed_validation_minibatch(*minibatch):
            self = __class__.get_instance()
            result = self.validate_minibatch(minibatch)
            result = self.process_validation_minibatch_result(minibatch, result)
            return result
        return _feed_validation_minibatch

    def process_validation_minibatch_result(self, minibatch, result):
        minibatch_size = len(minibatch[self.input_label_column_index])
        result = pd.DataFrame({
            'col0': self._to_pd_series(result[0], minibatch_size),
            'col1': self._to_pd_series(result[1], minibatch_size),
            'col2': self._to_pd_series(result[2], minibatch_size)
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
        metric = ESMMMetric()
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
            delta = ESMMMetric.from_states(states)
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

class ESMMMetric(ms.ModelMetric):
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
