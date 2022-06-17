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

#include <serving/sparse_lookup_model.h>

#include <memory>
#include <string>
#include <vector>

namespace metaspore {

template <typename K, typename V> class MemoryMappedArrayHashMap;

namespace serving {

class InMemorySparseLookupSource : public SparseLookupModel::SparseLookupSource {
  public:
    awaitable_status load(const std::string &dir) override;

    awaitable_result<std::shared_ptr<arrow::FloatTensor>>
    lookup(std::shared_ptr<arrow::UInt64Tensor> indices) override;

    awaitable_result<uint64_t> get_vector_size() override;

    static std::unique_ptr<SparseLookupModel::SparseLookupSource> make();

  private:
    std::vector<std::shared_ptr<MemoryMappedArrayHashMap<uint64_t, float>>> hashmaps_;
    uint64_t vector_size_ = 0;
};

} // namespace serving

} // namespace metaspore