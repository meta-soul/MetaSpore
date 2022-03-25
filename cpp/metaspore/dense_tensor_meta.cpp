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

#include <metaspore/dense_tensor_meta.h>
#include <metaspore/pybind_utils.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/tensor_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

void DenseTensorMeta::CheckDenseTensorMeta(int index) const {
    if (GetName().empty()) {
        std::string serr;
        serr.append("Can not compute dense tensor partition shapes, ");
        serr.append("as the name is empty.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    const int n = GetPartitionCount();
    if (n <= 0) {
        std::string serr;
        serr.append("Can not compute partition shapes for dense tensor '");
        serr.append(GetName());
        serr.append("', as the partition count ");
        serr.append(std::to_string(n));
        serr.append(" is invalid.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (index < 0 || index >= n) {
        std::string serr;
        serr.append("Can not compute partition shapes for dense tensor '");
        serr.append(GetName());
        serr.append("', as the partition index ");
        serr.append(std::to_string(index));
        serr.append(" is invalid; partition_count = ");
        serr.append(std::to_string(GetPartitionCount()));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (GetDataShape().empty()) {
        std::string serr;
        serr.append("Can not compute partition shapes for dense tensor '");
        serr.append(GetName());
        serr.append("', as the tensor data shape is empty; partition_count = ");
        serr.append(std::to_string(GetPartitionCount()));
        serr.append(", partition_index = ");
        serr.append(std::to_string(index));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    const size_t m = GetDataShape().at(0);
    if (m == 0) {
        std::string serr;
        serr.append("Can not compute partition shapes for dense tensor '");
        serr.append(GetName());
        serr.append("', as the tensor data shape [");
        for (size_t i = 0; i < GetDataShape().size(); i++) {
            serr.append(i ? ", " : "");
            serr.append(std::to_string(GetDataShape().at(i)));
        }
        serr.append("] is invalid; partition_count = ");
        serr.append(std::to_string(GetPartitionCount()));
        serr.append(", partition_index = ");
        serr.append(std::to_string(index));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (!GetStateShape().empty() && GetStateShape().at(0) != m) {
        std::string serr;
        serr.append("Can not compute partition shapes for dense tensor '");
        serr.append(GetName());
        serr.append("', as the tensor state shape [");
        for (size_t i = 0; i < GetStateShape().size(); i++) {
            serr.append(i ? ", " : "");
            serr.append(std::to_string(GetStateShape().at(i)));
        }
        serr.append("] is invalid; partition_count = ");
        serr.append(std::to_string(GetPartitionCount()));
        serr.append(", partition_index = ");
        serr.append(std::to_string(index));
        serr.append(", shape = [");
        for (size_t i = 0; i < GetDataShape().size(); i++) {
            serr.append(i ? ", " : "");
            serr.append(std::to_string(GetDataShape().at(i)));
        }
        serr.append("].\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
}

void DenseTensorMeta::ComputePartitionShapesWithHash(
    size_t hash, int index, size_t &begin, size_t &end, std::vector<size_t> *partition_data_shape,
    std::vector<size_t> *partition_state_shape) const {
    size_t m = GetDataShape().at(0);
    size_t n = GetPartitionCount();
    size_t i = index;
    size_t h = hash;
    i = (i + h) % n;
    size_t my_items = (m + n - 1 - i) / n;
    size_t avg_items = m / n;
    size_t remainder = m % n;
    begin = avg_items * i + std::min(i, remainder);
    end = begin + my_items;
    if (partition_data_shape) {
        *partition_data_shape = GetDataShape();
        partition_data_shape->at(0) = end - begin;
    }
    if (partition_state_shape) {
        *partition_state_shape = GetStateShape();
        if (!GetStateShape().empty())
            partition_state_shape->at(0) = end - begin;
    }
}

void DenseTensorMeta::ComputePartitionShapes(int index, size_t &begin, size_t &end,
                                             std::vector<size_t> *partition_data_shape,
                                             std::vector<size_t> *partition_state_shape) const {
    const size_t hash = GetNameHash();
    ComputePartitionShapesWithHash(hash, index, begin, end, partition_data_shape,
                                   partition_state_shape);
}

void DenseTensorMeta::SetInitializerByData(std::string data) {
    namespace py = pybind11;
    using namespace py::literals;
    if (data.empty()) {
        initializer_ = {};
        initializer_object_ = {};
    } else {
        py::gil_scoped_acquire gil;
        py::object obj = metaspore::deserialize_pyobject(data);
        MakeInitializerReady(obj);
        std::shared_ptr<py::object> func = metaspore::make_shared_pyobject(obj);
        initializer_ = [func](const std::string &name, metaspore::SmartArray<uint8_t> data,
                              const DenseTensorMeta &meta) {
            py::gil_scoped_acquire gil;
            py::array data_arr = metaspore::make_numpy_array(data, meta.data_type_);
            py::tuple data_shape(meta.data_shape_.size());
            for (size_t i = 0; i < meta.data_shape_.size(); i++)
                data_shape[i] = (i == 0) ? -1 : static_cast<int64_t>(meta.data_shape_.at(i));
            data_arr = data_arr.attr("reshape")(data_shape);
            (*func)("name"_a = name, "data"_a = data_arr, "keys"_a = py::none());
        };
        initializer_object_ = std::move(func);
    }
}

void DenseTensorMeta::SetUpdaterByData(std::string data) {
    namespace py = pybind11;
    using namespace py::literals;
    if (data.empty()) {
        updater_ = {};
        updater_object_ = {};
    } else {
        py::gil_scoped_acquire gil;
        py::object obj = metaspore::deserialize_pyobject(data);
        MakeUpdaterReady(obj);
        std::shared_ptr<py::object> func = metaspore::make_shared_pyobject(obj);
        updater_ = [func](const std::string &name, metaspore::SmartArray<uint8_t> param,
                          metaspore::SmartArray<uint8_t> grad, metaspore::SmartArray<uint8_t> state,
                          const DenseTensorMeta &meta) {
            // Some PyTorch operations such as ``grad.clone()`` and ``XXX + grad``
            // require memory alignment, we use ``SmartArray::Copy`` to use GLIBC allocated
            // memory which is 16 bytes aligned.
            metaspore::SmartArray<uint8_t> grad_clone = grad.Copy();
            py::gil_scoped_acquire gil;
            py::array param_arr = metaspore::make_numpy_array(param, meta.data_type_);
            py::array grad_arr = metaspore::make_numpy_array(grad_clone, meta.data_type_);
            py::tuple data_shape(meta.data_shape_.size());
            for (size_t i = 0; i < meta.data_shape_.size(); i++)
                data_shape[i] = (i == 0) ? -1 : static_cast<int64_t>(meta.data_shape_.at(i));
            param_arr = param_arr.attr("reshape")(data_shape);
            grad_arr = grad_arr.attr("reshape")(data_shape);
            if (meta.state_shape_.empty()) {
                (*func)("name"_a = name, "param"_a = param_arr, "grad"_a = grad_arr,
                        "state"_a = py::none(), "indices"_a = py::none(), "keys"_a = py::none());
            } else {
                py::array state_arr = metaspore::make_numpy_array(state, meta.data_type_);
                py::tuple state_shape(meta.state_shape_.size());
                for (size_t i = 0; i < meta.state_shape_.size(); i++)
                    state_shape[i] = (i == 0) ? -1 : static_cast<int64_t>(meta.state_shape_.at(i));
                state_arr = state_arr.attr("reshape")(state_shape);
                (*func)("name"_a = name, "param"_a = param_arr, "grad"_a = grad_arr,
                        "state"_a = state_arr, "indices"_a = py::none(), "keys"_a = py::none());
            }
        };
        updater_object_ = std::move(func);
    }
}

std::string DenseTensorMeta::GetInitializerAsData() const {
    pybind11::gil_scoped_acquire gil;
    if (!initializer_object_.has_value())
        return {};
    auto func = std::any_cast<std::shared_ptr<pybind11::object>>(initializer_object_);
    return metaspore::serialize_pyobject(*func);
}

std::string DenseTensorMeta::GetUpdaterAsData() const {
    pybind11::gil_scoped_acquire gil;
    if (!updater_object_.has_value())
        return {};
    auto func = std::any_cast<std::shared_ptr<pybind11::object>>(updater_object_);
    return metaspore::serialize_pyobject(*func);
}

std::string DenseTensorMeta::ToString() const { return ToJsonString(); }

std::string DenseTensorMeta::ToJsonString() const { return to_json().dump(); }

json11::Json DenseTensorMeta::to_json() const {
    return json11::Json::object{
        {"name", name_},
        {"data_type", NullableDataTypeToString(data_type_)},
        {"data_shape", ShapeToString(data_shape_)},
        {"state_shape", ShapeToString(state_shape_)},
        {"initializer_data", GetInitializerAsData()},
        {"updater_data", GetUpdaterAsData()},
        {"partition_count", partition_count_},
    };
}

DenseTensorMeta DenseTensorMeta::FromJsonString(const std::string &str) {
    std::string err;
    json11::Json json = json11::Json::parse(str, err);
    if (!err.empty()) {
        std::string serr;
        serr.append("Unable to create DenseTensorMeta from JSON string; str: ");
        serr.append(str);
        serr.append(", err: ");
        serr.append(err);
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    return FromJson(std::move(json));
}

DenseTensorMeta DenseTensorMeta::FromJson(json11::Json json) {
    DenseTensorMeta meta;
    meta.SetName(json["name"].string_value());
    meta.SetDataType(DataTypeFromString(json["data_type"].string_value()));
    meta.SetDataShape(ShapeFromString(json["data_shape"].string_value()));
    meta.SetStateShape(ShapeFromString(json["state_shape"].string_value()));
    meta.SetInitializerByData(json["initializer_data"].string_value());
    meta.SetUpdaterByData(json["updater_data"].string_value());
    meta.SetPartitionCount(json["partition_count"].int_value());
    return meta;
}

bool DenseTensorMeta::IsCompatible(const DenseTensorMeta &rhs) const {
    return data_type_ == rhs.data_type_ && data_shape_ == rhs.data_shape_ &&
           state_shape_ == rhs.state_shape_;
}

bool DenseTensorMeta::operator==(const DenseTensorMeta &rhs) const {
    return name_ == rhs.name_ && data_type_ == rhs.data_type_ && data_shape_ == rhs.data_shape_ &&
           state_shape_ == rhs.state_shape_ &&
           GetInitializerAsData() == rhs.GetInitializerAsData() &&
           GetUpdaterAsData() == rhs.GetUpdaterAsData() && partition_count_ == rhs.partition_count_;
}

} // namespace metaspore
