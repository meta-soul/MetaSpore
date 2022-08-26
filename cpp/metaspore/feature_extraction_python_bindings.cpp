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
#include <metaspore/feature_extraction_python_bindings.h>
#include <metaspore/sparse_feature_extractor.h>
#include <metaspore/pybind_utils.h>
#include <metaspore/stack_trace_utils.h>
#include <stdexcept>

namespace py = pybind11;

namespace metaspore {

void DefineFeatureExtractionBindings(pybind11::module &m) {
    auto status = metaspore::RegisterCustomArrowFunctions();
    if (!status.ok()) {
        std::string serr;
        serr.append("Fail to register custom arrow functions in MetaSpore extension.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }

    int rc = arrow::py::import_pyarrow();
    if (rc != 0) {
        std::string serr;
        serr.append("Fail to import pyarrow in MetaSpore extension.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }

    py::class_<metaspore::SparseFeatureExtractor, std::shared_ptr<metaspore::SparseFeatureExtractor>>(
        m, "SparseFeatureExtractor")
        .def_property_readonly("source_table_name", &metaspore::SparseFeatureExtractor::get_source_table_name)
        .def_property_readonly("schema_source", &metaspore::SparseFeatureExtractor::get_schema_source)
        .def_property_readonly("feature_count", &metaspore::SparseFeatureExtractor::get_feature_count)
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
        })
        .def(py::pickle(
            [](const metaspore::SparseFeatureExtractor &extractor) {
                auto &str1 = extractor.get_source_table_name();
                auto &str2 = extractor.get_schema_source();
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
                auto extractor = std::make_shared<metaspore::SparseFeatureExtractor>(str1, str2);
                return extractor;
        }));

    py::class_<metaspore::HashUniquifier>(m, "HashUniquifier")
        .def_static("uniquify", [](py::array_t<uint64_t> items) {
            std::vector<uint64_t> entries =
                HashUniquifier::Uniquify(items.mutable_data(), items.size());
            return metaspore::to_numpy_array(std::move(entries));
        });
}

} // namespace metaspore
