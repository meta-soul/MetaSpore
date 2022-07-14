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

#include <arrow/python/pyarrow.h>
#include <common/features/feature_compute_funcs.h>
#include <common/hashmap/hash_uniquifier.h>
#include <memory>
#include <metaspore/combine_schema.h>
#include <metaspore/feature_extraction_python_bindings.h>
#include <metaspore/index_batch.h>
#include <metaspore/sparse_feature_extractor.h>
#include <metaspore/pybind_utils.h>
#include <metaspore/stack_trace_utils.h>
#include <stdexcept>
#include <iostream>

#include <gflags/gflags.h>

namespace py = pybind11;

namespace metaspore {

void DefineFeatureExtractionBindings(pybind11::module &m) {
    // TODO: cf: change to use env vars to init the thread pool
    int argc = 1;
    char *arga[] = {"test", nullptr};
    char **argv = &arga[0];
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto status = metaspore::RegisterAllArrowFunctions();
    if (!status.ok()) {
        std::string serr;
        serr.append("Fail to register arrow functions in MetaSpore extension.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }

    const int rc = arrow::py::import_pyarrow();
    if (rc != 0) {
        std::string serr;
        serr.append("Fail to import pyarrow in MetaSpore extension.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }

    py::class_<metaspore::CombineSchema, std::shared_ptr<metaspore::CombineSchema>>(m,
                                                                                    "CombineSchema")
        .def_property_readonly("feature_count", &metaspore::CombineSchema::GetFeatureCount)
        .def_property_readonly("column_name_source", &metaspore::CombineSchema::GetColumnNameSource)
        .def_property_readonly("combine_schema_source",
                               &metaspore::CombineSchema::GetCombineSchemaSource)
        .def(py::init<>())
        .def("clear", &metaspore::CombineSchema::Clear)
        .def("load_column_name_from_source", &metaspore::CombineSchema::LoadColumnNameFromSource)
        .def("load_column_name_from_file", &metaspore::CombineSchema::LoadColumnNameFromFile)
        .def("load_combine_schema_from_source",
             &metaspore::CombineSchema::LoadCombineSchemaFromSource)
        .def("load_combine_schema_from_file", &metaspore::CombineSchema::LoadCombineSchemaFromFile)
        .def("get_column_name_map",
             [](const metaspore::CombineSchema &schema) {
                 py::dict map;
                 for (auto &&[key, value] : schema.GetColumnNameMap())
                     map[py::str(key)] = value;
                 return map;
             })
        .def("combine_to_indices_and_offsets",
             [](const metaspore::CombineSchema &schema, const metaspore::IndexBatch &batch,
                bool feature_offset) {
                 auto [indices, offsets] = schema.CombineToIndicesAndOffsets(batch, feature_offset);
                 py::array indices_arr = metaspore::to_numpy_array(std::move(indices));
                 py::array offsets_arr = metaspore::to_numpy_array(std::move(offsets));
                 return py::make_tuple(indices_arr, offsets_arr);
             })
        .def_static(
            "compute_feature_hash",
            [](py::object feature) {
                std::vector<std::pair<std::string, std::string>> vec;
                for (const auto &item : feature) {
                    const py::tuple t = item.cast<py::tuple>();
                    std::string name = t[0].cast<std::string>();
                    std::string value = t[1].cast<std::string>();
                    if (value == "none") {
                        std::string serr;
                        serr.append(
                            "none as value is invalid, because it should have been filtered\n\n");
                        serr.append(GetStackTrace());
                        spdlog::error(serr);
                        throw std::runtime_error(serr);
                    }
                    vec.emplace_back(std::move(name), std::move(value));
                }
                return metaspore::CombineSchema::ComputeFeatureHash(vec);
            })
        .def(py::pickle(
            [](const metaspore::CombineSchema &schema) {
                auto &str1 = schema.GetColumnNameSource();
                auto &str2 = schema.GetCombineSchemaSource();
                return py::make_tuple(str1, str2);
            },
            [](py::tuple t) {
                if (t.size() != 2) {
                    std::string serr;
                    serr.append("invalid pickle state\n\n");
                    serr.append(GetStackTrace());
                    spdlog::error(serr);
                    throw std::runtime_error(serr);
                }
                std::string str1 = t[0].cast<std::string>();
                std::string str2 = t[1].cast<std::string>();
                auto schema = std::make_shared<metaspore::CombineSchema>();
                schema->LoadColumnNameFromSource(str1);
                schema->LoadCombineSchemaFromSource(str2);
                return schema;
            }));

    py::class_<metaspore::IndexBatch, std::shared_ptr<metaspore::IndexBatch>>(m, "IndexBatch")
        .def_property_readonly("rows", &metaspore::IndexBatch::GetRows)
        .def_property_readonly("columns", &metaspore::IndexBatch::GetColumns)
        .def(py::init<py::list, const std::string &>())
        .def("to_list", &metaspore::IndexBatch::ToList)
        .def("__str__", &metaspore::IndexBatch::ToString);

    py::class_<metaspore::SparseFeatureExtractor, std::shared_ptr<metaspore::SparseFeatureExtractor>>(
        m, "SparseFeatureExtractor")
        .def_property_readonly("source_table_name", &metaspore::SparseFeatureExtractor::get_source_table_name)
        .def_property_readonly("schema_source", &metaspore::SparseFeatureExtractor::get_schema_source)
        .def(py::init<const std::string &, const std::string &>())
        .def("extract", [](metaspore::SparseFeatureExtractor &extractor, py::object batch) {
            auto result = arrow::py::unwrap_batch(batch.ptr());
            if (!result.ok()) {
                std::string serr;
                serr.append("Unable to unwrap arrow record batch. ");
                serr.append(result.status().message());
                serr.append("\n\n");
                serr.append(GetStackTrace());
                spdlog::error(serr);
                throw std::runtime_error(serr);
            }
            auto unwrapped_batch = *result;
            auto [indices, offsets] = extractor.extract(unwrapped_batch);
            py::array indices_arr = metaspore::to_numpy_array(std::move(indices));
            py::array offsets_arr = metaspore::to_numpy_array(std::move(offsets));
            return py::make_tuple(indices_arr, offsets_arr);
        });

    py::class_<metaspore::HashUniquifier>(m, "HashUniquifier")
        .def_static("uniquify", [](py::array_t<uint64_t> items) {
            std::vector<uint64_t> entries =
                HashUniquifier::Uniquify(items.mutable_data(), items.size());
            return metaspore::to_numpy_array(std::move(entries));
        });
}

} // namespace metaspore
