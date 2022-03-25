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

#include <metaspore/pybind_utils.h>
#include <metaspore/sparse_tensor_meta.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/tensor_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

void SparseTensorMeta::CheckSparseTensorMeta(int index) const {
    if (GetName().empty()) {
        std::string serr;
        serr.append("Can not compute sparse tensor slice support info, ");
        serr.append("as the name is empty.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    const int n = GetPartitionCount();
    if (n <= 0) {
        std::string serr;
        serr.append("Can not compute slice support info for sparse tensor '");
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
        serr.append("Can not compute slice support info for sparse tensor '");
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
}

void SparseTensorMeta::ComputeSliceInfo() {
    const size_t item_size = DataTypeToSize(data_type_);
    slice_data_length_ = item_size * TotalElements(slice_data_shape_);
    slice_state_length_ = item_size * TotalElements(slice_state_shape_);
    const size_t age_offset = slice_data_length_ + slice_state_length_;
    const size_t age_size = std::max(item_size, sizeof(int));
    const size_t age_mask = age_size - 1;
    slice_age_offset_ = (age_offset + age_mask) & ~age_mask;
    slice_total_bytes_ = slice_age_offset_ + age_size;
}

void SparseTensorMeta::SetInitializerByData(std::string data) {
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
                              metaspore::SmartArray<uint8_t> keys, const SparseTensorMeta &meta) {
            {
                py::gil_scoped_acquire gil;
                const size_t item_size = metaspore::DataTypeToSize(meta.data_type_);
                const size_t cols = meta.slice_total_bytes_ / item_size;
                const size_t data_cols = meta.slice_data_length_ / item_size;
                py::array blob_arr = metaspore::make_numpy_array(data, meta.data_type_);
                blob_arr = blob_arr.attr("reshape")(-1, cols);
                py::slice data_slice(0, data_cols, 1);
                py::tuple data_subscript = py::make_tuple(py::ellipsis(), data_slice);
                py::array data_arr = blob_arr[data_subscript];
                py::tuple data_shape(1 + meta.slice_data_shape_.size());
                data_shape[0] = -1;
                for (size_t i = 0; i < meta.slice_data_shape_.size(); i++)
                    data_shape[1 + i] = static_cast<int64_t>(meta.slice_data_shape_.at(i));
                data_arr = data_arr.attr("reshape")(data_shape);
                py::array keys_arr = metaspore::make_numpy_array(keys, metaspore::DataType::UInt64);
                (*func)("name"_a = name, "data"_a = data_arr, "keys"_a = keys_arr);
            }
            const size_t slice_count = data.size() / meta.slice_total_bytes_;
            uint8_t *slice_ptr = data.data();
            for (size_t i = 0; i < slice_count; i++) {
                memset(slice_ptr + meta.slice_data_length_, 0,
                       meta.slice_total_bytes_ - meta.slice_data_length_);
                slice_ptr += meta.slice_total_bytes_;
            }
        };
        initializer_object_ = std::move(func);
    }
}

void SparseTensorMeta::SetUpdaterByData(std::string data) {
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
                          metaspore::SmartArray<uint8_t> grad,
                          metaspore::SmartArray<uint8_t> indices,
                          metaspore::SmartArray<uint8_t> keys, const SparseTensorMeta &meta) {
            // Some PyTorch operations such as ``grad.clone()`` and ``XXX + grad``
            // require memory alignment, we use ``SmartArray::Copy`` to use GLIBC allocated
            // memory which is 16 bytes aligned.
            metaspore::SmartArray<uint8_t> grad_clone = grad.Copy();
            py::gil_scoped_acquire gil;
            const size_t item_size = metaspore::DataTypeToSize(meta.data_type_);
            const size_t cols = meta.slice_total_bytes_ / item_size;
            const size_t data_cols = meta.slice_data_length_ / item_size;
            py::array blob_arr = metaspore::make_numpy_array(param, meta.data_type_);
            blob_arr = blob_arr.attr("reshape")(-1, cols);
            py::slice data_slice(0, data_cols, 1);
            py::tuple data_subscript = py::make_tuple(py::ellipsis(), data_slice);
            py::array data_arr = blob_arr[data_subscript];
            py::tuple data_shape(1 + meta.slice_data_shape_.size());
            data_shape[0] = -1;
            for (size_t i = 0; i < meta.slice_data_shape_.size(); i++)
                data_shape[1 + i] = static_cast<int64_t>(meta.slice_data_shape_.at(i));
            data_arr = data_arr.attr("reshape")(data_shape);
            py::array grad_arr = metaspore::make_numpy_array(grad_clone, meta.data_type_);
            grad_arr = grad_arr.attr("reshape")(data_shape);
            py::array indices_arr =
                metaspore::make_numpy_array(indices, metaspore::DataType::UInt64);
            py::array keys_arr = metaspore::make_numpy_array(keys, metaspore::DataType::UInt64);
            if (meta.slice_state_shape_.empty()) {
                (*func)("name"_a = name, "param"_a = data_arr, "grad"_a = grad_arr,
                        "state"_a = py::none(), "indices"_a = indices_arr, "keys"_a = keys_arr);
            } else {
                const size_t state_cols = meta.slice_state_length_ / item_size;
                py::slice state_slice(data_cols, data_cols + state_cols, 1);
                py::tuple state_subscript = py::make_tuple(py::ellipsis(), state_slice);
                py::array state_arr = blob_arr[state_subscript];
                py::tuple state_shape(1 + meta.slice_state_shape_.size());
                state_shape[0] = -1;
                for (size_t i = 0; i < meta.slice_state_shape_.size(); i++)
                    state_shape[1 + i] = static_cast<int64_t>(meta.slice_state_shape_.at(i));
                state_arr = state_arr.attr("reshape")(state_shape);
                (*func)("name"_a = name, "param"_a = data_arr, "grad"_a = grad_arr,
                        "state"_a = state_arr, "indices"_a = indices_arr, "keys"_a = keys_arr);
            }
        };
        updater_object_ = std::move(func);
    }
}

std::string SparseTensorMeta::GetInitializerAsData() const {
    pybind11::gil_scoped_acquire gil;
    if (!initializer_object_.has_value())
        return {};
    auto func = std::any_cast<std::shared_ptr<pybind11::object>>(initializer_object_);
    return metaspore::serialize_pyobject(*func);
}

std::string SparseTensorMeta::GetUpdaterAsData() const {
    pybind11::gil_scoped_acquire gil;
    if (!updater_object_.has_value())
        return {};
    auto func = std::any_cast<std::shared_ptr<pybind11::object>>(updater_object_);
    return metaspore::serialize_pyobject(*func);
}

std::string SparseTensorMeta::ToString() const { return ToJsonString(); }

std::string SparseTensorMeta::ToJsonString() const { return to_json().dump(); }

json11::Json SparseTensorMeta::to_json() const {
    return json11::Json::object{
        {"name", name_},
        {"data_type", NullableDataTypeToString(data_type_)},
        {"slice_data_shape", ShapeToString(slice_data_shape_)},
        {"slice_state_shape", ShapeToString(slice_state_shape_)},
        {"initializer_data", GetInitializerAsData()},
        {"updater_data", GetUpdaterAsData()},
        {"partition_count", partition_count_},
    };
}

SparseTensorMeta SparseTensorMeta::FromJsonString(const std::string &str) {
    std::string err;
    json11::Json json = json11::Json::parse(str, err);
    if (!err.empty()) {
        std::string serr;
        serr.append("Unable to create SparseTensorMeta from JSON string; str: ");
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

SparseTensorMeta SparseTensorMeta::FromJson(json11::Json json) {
    SparseTensorMeta meta;
    meta.SetName(json["name"].string_value());
    meta.SetDataType(DataTypeFromString(json["data_type"].string_value()));
    meta.SetSliceDataShape(ShapeFromString(json["slice_data_shape"].string_value()));
    meta.SetSliceStateShape(ShapeFromString(json["slice_state_shape"].string_value()));
    meta.SetInitializerByData(json["initializer_data"].string_value());
    meta.SetUpdaterByData(json["updater_data"].string_value());
    meta.SetPartitionCount(json["partition_count"].int_value());
    meta.ComputeSliceInfo();
    return meta;
}

bool SparseTensorMeta::IsCompatible(const SparseTensorMeta &rhs) const {
    return data_type_ == rhs.data_type_ && slice_data_shape_ == rhs.slice_data_shape_ &&
           slice_state_shape_ == rhs.slice_state_shape_;
}

bool SparseTensorMeta::IsCompatibleRelaxed(const SparseTensorMeta &rhs, bool data_only) const {
    if (!data_only)
        return IsCompatible(rhs);
    return data_type_ == rhs.data_type_ && slice_data_shape_ == rhs.slice_data_shape_;
}

bool SparseTensorMeta::operator==(const SparseTensorMeta &rhs) const {
    return name_ == rhs.name_ && data_type_ == rhs.data_type_ &&
           slice_data_shape_ == rhs.slice_data_shape_ &&
           slice_state_shape_ == rhs.slice_state_shape_ &&
           GetInitializerAsData() == rhs.GetInitializerAsData() &&
           GetUpdaterAsData() == rhs.GetUpdaterAsData() && partition_count_ == rhs.partition_count_;
}

} // namespace metaspore
