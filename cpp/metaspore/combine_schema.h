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

#include <metaspore/index_batch.h>
#include <metaspore/string_utils.h>
#include <sstream>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace metaspore {

class CombineSchema {
  public:
    void Clear();

    void LoadColumnNameFromStream(std::istream &stream);
    void LoadColumnNameFromSource(const std::string &source);
    void LoadColumnNameFromFile(const std::string &uri);

    void LoadCombineSchemaFromStream(std::istream &stream);
    void LoadCombineSchemaFromSource(const std::string &source);
    void LoadCombineSchemaFromFile(const std::string &uri);

    size_t GetFeatureCount() const { return combine_columns_.size(); }

    const std::string &GetColumnNameSource() const { return column_name_source_; }
    const std::string &GetCombineSchemaSource() const { return combine_schema_source_; }
    const std::unordered_map<std::string, int> &GetColumnNameMap() const {
        return column_name_map_;
    }

    std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>
    CombineToIndicesAndOffsets(const IndexBatch &batch, bool feature_offset) const;

    static uint64_t
    ComputeFeatureHash(const std::vector<std::pair<std::string, std::string>> &feature);

  private:
    static constexpr uint64_t CombineOneField(uint64_t name, uint64_t value) {
        return CombineHashCodes(name, value);
    }

    static constexpr uint64_t ConcatOneField(uint64_t first, uint64_t second) {
        return CombineHashCodes(first, second);
    }

    static void CombineOneFeature(const std::vector<const StringViewHashVector *> &splits,
                                  const std::vector<std::string> &names,
                                  const std::vector<uint64_t> &name_hashes,
                                  std::vector<uint64_t> &combine_hashes, size_t total_results);

    const StringViewHashVector *GetCell(const IndexBatch &batch, size_t i,
                                        const std::string &column_name) const;

    std::unordered_map<std::string, int> column_name_map_;
    std::vector<std::vector<std::string>> combine_columns_;
    std::vector<std::vector<std::string>> combine_columns_aliases_;
    std::vector<std::vector<uint64_t>> combine_columns_aliases_hashes_;
    std::vector<std::string> column_names_;
    std::string column_name_source_;
    std::string combine_schema_source_;
};

} // namespace metaspore
