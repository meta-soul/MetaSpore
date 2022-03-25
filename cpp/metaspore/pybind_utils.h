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

#include <memory>
#include <metaspore/smart_array.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace metaspore {

std::shared_ptr<pybind11::object> make_shared_pyobject(pybind11::object obj);

template <typename T> std::shared_ptr<T> extract_shared_pyobject(pybind11::object obj) {
    std::shared_ptr<pybind11::object> obj_ptr = make_shared_pyobject(std::move(obj));
    std::shared_ptr<T> ptr1 = obj_ptr->cast<std::shared_ptr<T>>();
    std::shared_ptr<T> ptr2(obj_ptr, ptr1.get());
    return std::move(ptr2);
}

std::string serialize_pyobject(pybind11::object obj);
pybind11::object deserialize_pyobject(const std::string &data);

void fixup_attributes(pybind11::object obj);

pybind11::array make_numpy_array(SmartArray<uint8_t> data, DataType dtype);

template <typename T> inline pybind11::array make_numpy_array(SmartArray<T> data) {
    SmartArray<uint8_t> data_u8 = data.template Cast<uint8_t>();
    DataType dtype = DataTypeToCode<T>::value;
    return make_numpy_array(data_u8, dtype);
}

template <typename T> inline pybind11::array to_numpy_array(std::vector<T> data) {
    auto data_arr = SmartArray<T>::Wrap(std::move(data));
    return make_numpy_array(data_arr);
}

template <typename T> pybind11::tuple make_python_tuple(const std::vector<T> &vec) {
    pybind11::list result(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        result[i] = vec.at(i);
    return result;
}

template <typename T> pybind11::list make_python_list(const std::vector<T> &vec) {
    pybind11::list result(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        result[i] = vec.at(i);
    return result;
}

template <typename T> std::vector<T> make_cpp_vector(pybind11::object obj) {
    std::vector<T> result;
    for (pybind11::handle item : obj) {
        T t = item.cast<T>();
        result.push_back(std::move(t));
    }
    return result;
}

std::tuple<std::string_view, pybind11::bytes> make_string_object_tuple(pybind11::bytes obj);
std::tuple<std::string_view, pybind11::object> get_string_object_tuple(pybind11::object obj);

} // namespace metaspore
