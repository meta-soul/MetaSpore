//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <metaspore/model_metric_buffer.h>

namespace metaspore {

void ModelMetricBuffer::UpdateBuffer(pybind11::array_t<double> positive_buffer,
                                     pybind11::array_t<double> negative_buffer,
                                     pybind11::array_t<float> predictions,
                                     pybind11::array_t<float> labels) {
    const size_t buffer_size = positive_buffer.size();
    const size_t instance_count = labels.size();
    double *const pos_buf = positive_buffer.mutable_data();
    double *const neg_buf = negative_buffer.mutable_data();
    const float *const preds = predictions.data();
    const float *const labs = labels.data();
    for (size_t i = 0; i < instance_count; i++) {
        const float pred = preds[i];
        const float lab = labs[i];
        const int64_t bucket = static_cast<int64_t>(pred * (buffer_size - 1));
        if (lab > 0.0)
            pos_buf[bucket] += lab;
        else if (lab < 0.0)
            neg_buf[bucket] += -lab;
        else
            neg_buf[bucket] += 1.0;
    }
}

double ModelMetricBuffer::ComputeAUC(pybind11::array_t<double> positive_buffer,
                                     pybind11::array_t<double> negative_buffer) {
    const size_t buffer_size = positive_buffer.size();
    const double *const pos_buf = positive_buffer.mutable_data();
    const double *const neg_buf = negative_buffer.mutable_data();
    double auc = 0.0;
    double prev_pos_sum = 0;
    double pos_sum = 0;
    double neg_sum = 0;
    for (size_t i = 0; i < buffer_size; i++) {
        const double pos = pos_buf[i];
        const double neg = neg_buf[i];
        prev_pos_sum = pos_sum;
        pos_sum += pos;
        neg_sum += neg;
        auc += 0.5 * (prev_pos_sum + pos_sum) * neg;
    }
    if (pos_sum == 0)
        auc = (neg_sum > 0) ? 0.0 : 1.0;
    else if (neg_sum == 0)
        auc = 1.0;
    else
        auc = 1.0 - auc / pos_sum / neg_sum;
    return auc;
}

} // namespace metaspore
