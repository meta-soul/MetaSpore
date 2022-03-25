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

#include <limits>
#include <metaspore/pybind_utils.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/tensor_utils.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>

namespace metaspore {

size_t SliceElements(const std::vector<size_t> &shape) {
    if (shape.empty())
        return 0;
    size_t n = 1;
    for (size_t i = 1; i < shape.size(); i++)
        n *= shape[i];
    return n;
}

size_t TotalElements(const std::vector<size_t> &shape) {
    if (shape.empty())
        return 0;
    size_t n = 1;
    for (size_t i = 0; i < shape.size(); i++)
        n *= shape[i];
    return n;
}

std::string ShapeToString(const std::vector<size_t> &shape) {
    std::ostringstream sout;
    for (size_t i = 0; i < shape.size(); i++)
        sout << (i ? " " : "") << shape.at(i);
    return sout.str();
}

std::vector<size_t> ShapeFromString(const std::string &str) {
    std::vector<size_t> shape;
    std::istringstream sin(str);
    size_t dim;
    while (sin >> dim)
        shape.push_back(dim);
    return shape;
}

template <typename T> void FillNaNValues(uint8_t *buffer, size_t size) {
    if (size % sizeof(T) != 0) {
        std::string serr;
        serr.append("Buffer size ");
        serr.append(std::to_string(size));
        serr.append(" is not a multiple of sizeof(");
        serr.append(DataTypeToString(DataTypeToCode<T>::value));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    T *buf = reinterpret_cast<T *>(buffer);
    const size_t n = size / sizeof(T);
    for (size_t i = 0; i < n; i++)
        buf[i] = std::numeric_limits<T>::quiet_NaN();
}

void FillNaN(uint8_t *buffer, size_t size, DataType type) {
    switch (type) {
    case DataType::Float32:
        FillNaNValues<float>(buffer, size);
        break;
    case DataType::Float64:
        FillNaNValues<double>(buffer, size);
        break;
    default:
        std::string serr;
        serr.append("DataType must be float32 or float64 to fill NaN values; ");
        serr.append(DataTypeToString(type));
        serr.append(" is invalid.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
}

void MakeInitializerReady(pybind11::object initializer) { fixup_attributes(initializer); }

void MakeUpdaterReady(pybind11::object updater) { fixup_attributes(updater); }

} // namespace metaspore
