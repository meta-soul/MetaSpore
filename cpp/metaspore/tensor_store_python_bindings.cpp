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

#include <metaspore/dense_tensor.h>
#include <metaspore/ps_agent.h>
#include <metaspore/pybind_utils.h>
#include <metaspore/sparse_tensor.h>
#include <metaspore/tensor_store_python_bindings.h>

namespace py = pybind11;

namespace metaspore {

void DefineTensorStoreBindings(pybind11::module &m) {
    py::class_<metaspore::DenseTensor>(m, "DenseTensor")
        .def(py::init<>())
        .def_property(
            "name", [](const metaspore::DenseTensor &self) { return self.GetMeta().GetName(); },
            [](metaspore::DenseTensor &self, std::string value) {
                self.GetMeta().SetName(std::move(value));
            })
        .def_property(
            "data_type",
            [](const metaspore::DenseTensor &self) {
                const metaspore::DataType t = self.GetMeta().GetDataType();
                return metaspore::NullableDataTypeToString(t);
            },
            [](metaspore::DenseTensor &self, const std::string &value) {
                const metaspore::DataType t = metaspore::NullableDataTypeFromString(value);
                self.GetMeta().SetDataType(t);
            })
        .def_property(
            "data_shape",
            [](const metaspore::DenseTensor &self) {
                const std::vector<size_t> &shape = self.GetMeta().GetDataShape();
                return metaspore::make_python_tuple(shape);
            },
            [](metaspore::DenseTensor &self, py::tuple value) {
                std::vector<size_t> shape = metaspore::make_cpp_vector<size_t>(value);
                self.GetMeta().SetDataShape(std::move(shape));
            })
        .def_property(
            "state_shape",
            [](const metaspore::DenseTensor &self) {
                const std::vector<size_t> &shape = self.GetMeta().GetStateShape();
                return metaspore::make_python_tuple(shape);
            },
            [](metaspore::DenseTensor &self, py::tuple value) {
                std::vector<size_t> shape = metaspore::make_cpp_vector<size_t>(value);
                self.GetMeta().SetStateShape(std::move(shape));
            })
        .def_property(
            "initializer",
            [](const metaspore::DenseTensor &self) {
                const std::string &data = self.GetMeta().GetInitializerAsData();
                return metaspore::deserialize_pyobject(data);
            },
            [](metaspore::DenseTensor &self, py::object value) {
                std::string data = metaspore::serialize_pyobject(value);
                self.GetMeta().SetInitializerByData(std::move(data));
            })
        .def_property(
            "updater",
            [](const metaspore::DenseTensor &self) {
                const std::string &data = self.GetMeta().GetUpdaterAsData();
                return metaspore::deserialize_pyobject(data);
            },
            [](metaspore::DenseTensor &self, py::object value) {
                std::string data = metaspore::serialize_pyobject(value);
                self.GetMeta().SetUpdaterByData(std::move(data));
            })
        .def_property(
            "partition_count",
            [](const metaspore::DenseTensor &self) { return self.GetMeta().GetPartitionCount(); },
            [](metaspore::DenseTensor &self, int value) {
                self.GetMeta().SetPartitionCount(value);
            })
        .def_property("agent", &metaspore::DenseTensor::GetAgent, &metaspore::DenseTensor::SetAgent)
        .def("__str__",
             [](const metaspore::DenseTensor &self) { return self.GetMeta().ToString(); })
        .def("init",
             [](metaspore::DenseTensor &self, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Init([func]() {
                     py::gil_scoped_acquire gil;
                     (*func)();
                 });
             })
        .def("dispose",
             [](metaspore::DenseTensor &self, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Dispose([func]() {
                     py::gil_scoped_acquire gil;
                     (*func)();
                 });
             })
        .def("push",
             [](metaspore::DenseTensor &self, py::array in, py::object cb, bool is_value,
                bool is_state) {
                 auto in_obj = metaspore::make_shared_pyobject(in);
                 void *in_data_ptr = const_cast<void *>(in.data(0));
                 uint8_t *in_data = static_cast<uint8_t *>(in_data_ptr);
                 auto in_array = metaspore::SmartArray<uint8_t>::Create(in_data, in.nbytes(),
                                                                        [in_obj](uint8_t *) {});
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Push(
                     in_array,
                     [func]() {
                         py::gil_scoped_acquire gil;
                         (*func)();
                     },
                     is_value, is_state);
             })
        .def("pull",
             [](metaspore::DenseTensor &self, py::object cb, bool is_state) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Pull(
                     [func, &self](metaspore::SmartArray<uint8_t> out) {
                         py::gil_scoped_acquire gil;
                         metaspore::DataType type = self.GetMeta().GetDataType();
                         py::object out_arr = metaspore::make_numpy_array(out, type);
                         py::tuple shape =
                             metaspore::make_python_tuple(self.GetMeta().GetDataShape());
                         out_arr = out_arr.attr("reshape")(shape);
                         (*func)(out_arr);
                     },
                     is_state);
             })
        .def("load",
             [](metaspore::DenseTensor &self, const std::string &dir_path, py::object cb,
                bool keep_meta) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Load(
                     dir_path,
                     [func]() {
                         py::gil_scoped_acquire gil;
                         (*func)();
                     },
                     keep_meta);
             })
        .def("save", [](metaspore::DenseTensor &self, const std::string &dir_path, py::object cb) {
            auto func = metaspore::make_shared_pyobject(cb);
            py::gil_scoped_release gil;
            self.Save(dir_path, [func]() {
                py::gil_scoped_acquire gil;
                (*func)();
            });
        });

    py::class_<metaspore::SparseTensor>(m, "SparseTensor")
        .def(py::init<>())
        .def_property(
            "name", [](const metaspore::SparseTensor &self) { return self.GetMeta().GetName(); },
            [](metaspore::SparseTensor &self, std::string value) {
                self.GetMeta().SetName(std::move(value));
            })
        .def_property(
            "data_type",
            [](const metaspore::SparseTensor &self) {
                const metaspore::DataType t = self.GetMeta().GetDataType();
                return metaspore::NullableDataTypeToString(t);
            },
            [](metaspore::SparseTensor &self, const std::string &value) {
                const metaspore::DataType t = metaspore::NullableDataTypeFromString(value);
                self.GetMeta().SetDataType(t);
            })
        .def_property(
            "slice_data_shape",
            [](const metaspore::SparseTensor &self) {
                const std::vector<size_t> &shape = self.GetMeta().GetSliceDataShape();
                return metaspore::make_python_tuple(shape);
            },
            [](metaspore::SparseTensor &self, py::tuple value) {
                std::vector<size_t> shape = metaspore::make_cpp_vector<size_t>(value);
                self.GetMeta().SetSliceDataShape(std::move(shape));
            })
        .def_property(
            "slice_state_shape",
            [](const metaspore::SparseTensor &self) {
                const std::vector<size_t> &shape = self.GetMeta().GetSliceStateShape();
                return metaspore::make_python_tuple(shape);
            },
            [](metaspore::SparseTensor &self, py::tuple value) {
                std::vector<size_t> shape = metaspore::make_cpp_vector<size_t>(value);
                self.GetMeta().SetSliceStateShape(std::move(shape));
            })
        .def_property(
            "initializer",
            [](const metaspore::SparseTensor &self) {
                const std::string &data = self.GetMeta().GetInitializerAsData();
                return metaspore::deserialize_pyobject(data);
            },
            [](metaspore::SparseTensor &self, py::object value) {
                std::string data = metaspore::serialize_pyobject(value);
                self.GetMeta().SetInitializerByData(std::move(data));
            })
        .def_property(
            "updater",
            [](const metaspore::SparseTensor &self) {
                const std::string &data = self.GetMeta().GetUpdaterAsData();
                return metaspore::deserialize_pyobject(data);
            },
            [](metaspore::SparseTensor &self, py::object value) {
                std::string data = metaspore::serialize_pyobject(value);
                self.GetMeta().SetUpdaterByData(std::move(data));
            })
        .def_property(
            "partition_count",
            [](const metaspore::SparseTensor &self) { return self.GetMeta().GetPartitionCount(); },
            [](metaspore::SparseTensor &self, int value) {
                self.GetMeta().SetPartitionCount(value);
            })
        .def_property("agent", &metaspore::SparseTensor::GetAgent,
                      &metaspore::SparseTensor::SetAgent)
        .def("__str__",
             [](const metaspore::SparseTensor &self) { return self.GetMeta().ToString(); })
        .def("init",
             [](metaspore::SparseTensor &self, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Init([func]() {
                     py::gil_scoped_acquire gil;
                     (*func)();
                 });
             })
        .def("dispose",
             [](metaspore::SparseTensor &self, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Dispose([func]() {
                     py::gil_scoped_acquire gil;
                     (*func)();
                 });
             })
        .def("clear",
             [](metaspore::SparseTensor &self, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Clear([func]() {
                     py::gil_scoped_acquire gil;
                     (*func)();
                 });
             })
        .def("push",
             [](metaspore::SparseTensor &self, py::array keys, py::array in, py::object cb,
                bool is_value) {
                 auto keys_obj = metaspore::make_shared_pyobject(keys);
                 auto in_obj = metaspore::make_shared_pyobject(in);
                 void *keys_data_ptr = const_cast<void *>(keys.data(0));
                 void *in_data_ptr = const_cast<void *>(in.data(0));
                 uint8_t *keys_data = static_cast<uint8_t *>(keys_data_ptr);
                 uint8_t *in_data = static_cast<uint8_t *>(in_data_ptr);
                 auto keys_array = metaspore::SmartArray<uint8_t>::Create(keys_data, keys.nbytes(),
                                                                          [keys_obj](uint8_t *) {});
                 auto in_array = metaspore::SmartArray<uint8_t>::Create(in_data, in.nbytes(),
                                                                        [in_obj](uint8_t *) {});
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Push(
                     keys_array, in_array,
                     [func]() {
                         py::gil_scoped_acquire gil;
                         (*func)();
                     },
                     is_value);
             })
        .def("pull",
             [](metaspore::SparseTensor &self, py::array keys, py::object cb, bool read_only,
                bool nan_fill) {
                 auto keys_obj = metaspore::make_shared_pyobject(keys);
                 void *keys_data_ptr = const_cast<void *>(keys.data(0));
                 uint8_t *keys_data = static_cast<uint8_t *>(keys_data_ptr);
                 auto keys_array = metaspore::SmartArray<uint8_t>::Create(keys_data, keys.nbytes(),
                                                                          [keys_obj](uint8_t *) {});
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Pull(
                     keys_array,
                     [func, &self](metaspore::SmartArray<uint8_t> out) {
                         py::gil_scoped_acquire gil;
                         metaspore::DataType type = self.GetMeta().GetDataType();
                         py::object out_arr = metaspore::make_numpy_array(out, type);
                         const std::vector<size_t> &slice_shape =
                             self.GetMeta().GetSliceDataShape();
                         py::tuple shape(1 + slice_shape.size());
                         shape[0] = -1;
                         for (size_t i = 0; i < slice_shape.size(); i++)
                             shape[1 + i] = static_cast<int64_t>(slice_shape.at(i));
                         out_arr = out_arr.attr("reshape")(shape);
                         (*func)(out_arr);
                     },
                     read_only, nan_fill);
             })
        .def("load",
             [](metaspore::SparseTensor &self, const std::string &dir_path, py::object cb,
                bool keep_meta) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Load(
                     dir_path,
                     [func]() {
                         py::gil_scoped_acquire gil;
                         (*func)();
                     },
                     keep_meta);
             })
        .def("save",
             [](metaspore::SparseTensor &self, const std::string &dir_path, py::object cb,
                bool text_mode) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Save(
                     dir_path,
                     [func]() {
                         py::gil_scoped_acquire gil;
                         (*func)();
                     },
                     text_mode);
             })
        .def("export",
             [](metaspore::SparseTensor &self, const std::string &dir_path, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.Export(dir_path, [func]() {
                     py::gil_scoped_acquire gil;
                     (*func)();
                 });
             })
        .def("import_from",
             [](metaspore::SparseTensor &self, const std::string &meta_file_path, py::object cb,
                bool data_only, bool skip_existing, bool transform_key,
                const std::string &feature_name) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.ImportFrom(
                     meta_file_path,
                     [func]() {
                         py::gil_scoped_acquire gil;
                         (*func)();
                     },
                     data_only, skip_existing, transform_key, feature_name);
             })
        .def("prune_small",
             [](metaspore::SparseTensor &self, double epsilon, py::object cb) {
                 auto func = metaspore::make_shared_pyobject(cb);
                 py::gil_scoped_release gil;
                 self.PruneSmall(epsilon, [func]() {
                     py::gil_scoped_acquire gil;
                     (*func)();
                 });
             })
        .def("prune_old", [](metaspore::SparseTensor &self, int max_age, py::object cb) {
            auto func = metaspore::make_shared_pyobject(cb);
            py::gil_scoped_release gil;
            self.PruneOld(max_age, [func]() {
                py::gil_scoped_acquire gil;
                (*func)();
            });
        });

    py::class_<metaspore::PSDefaultAgent, metaspore::PyPSDefaultAgent<>,
               std::shared_ptr<metaspore::PSDefaultAgent>, metaspore::PSAgent>(m, "PSDefaultAgent")
        .def(py::init<>())
        .def_property("py_agent", &metaspore::PSDefaultAgent::GetPyAgent,
                      &metaspore::PSDefaultAgent::SetPyAgent)
        .def("run", &metaspore::PSDefaultAgent::Run)
        .def("handle_request", &metaspore::PSDefaultAgent::HandleRequest)
        .def("finalize", &metaspore::PSDefaultAgent::Finalize);
}

} // namespace metaspore
