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

#include <arrow/tensor.h>
#include <boost/core/demangle.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <onnxruntime_cxx_api.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

namespace metaspore {

class TensorPrint {
  public:
    template <typename T> static void print_tensor(const arrow::Tensor &tensor) {
        xt::print_options::set_line_width(63);
        xt::print_options::set_precision(5);
        xt::print_options::set_threshold(1000000);
        auto array = xt::adapt((const T *)tensor.raw_data(), tensor.shape());
        std::cout << array << std::endl;
        fmt::print("shape [{}], type {}\n", fmt::join(tensor.shape(), ","),
                   boost::core::demangle(typeid(T).name()));
    }

    template <typename T> static void print_tensor(const Ort::Value &tensor) {
        xt::print_options::set_line_width(63);
        xt::print_options::set_precision(5);
        xt::print_options::set_threshold(1000000);
        auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto array = xt::adapt(tensor.GetTensorData<T>(), shape);
        std::cout << array << std::endl;
        fmt::print("shape [{}], type {}\n", fmt::join(shape, ","),
                   boost::core::demangle(typeid(T).name()));
    }
};

} // namespace metaspore
