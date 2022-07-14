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

#include <common/features/feature_compute_exec.h>

namespace metaspore {

class SparseFeatureExtractor {
public:
    SparseFeatureExtractor(const std::string &source_table_name,
                           const std::string &schema_source);

    std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>
    extract(std::shared_ptr<arrow::RecordBatch> batch);

    const std::string &get_source_table_name() const { return source_table_name_; }
    const std::string &get_schema_source() const { return schema_source_; }

private:
    void check_construct(const status &the_status);
    void check_extract(const status &the_status);
    void check_status(const status &the_status, const std::string &message);

    std::string source_table_name_;
    std::string schema_source_;
    FeatureComputeExec executor_;
};

} // namespace metaspore
