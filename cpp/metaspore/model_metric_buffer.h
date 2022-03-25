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

#pragma once

#include <pybind11/numpy.h>
#include <stdint.h>

namespace metaspore {

class ModelMetricBuffer {
  public:
    static void UpdateBuffer(pybind11::array_t<double> positive_buffer,
                             pybind11::array_t<double> negative_buffer,
                             pybind11::array_t<float> predictions, pybind11::array_t<float> labels);

    static double ComputeAUC(pybind11::array_t<double> positive_buffer,
                             pybind11::array_t<double> negative_buffer);
};

} // namespace metaspore
