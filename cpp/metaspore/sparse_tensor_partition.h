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

#include <common/hashmap/array_hash_map.h>
#include <metaspore/sparse_tensor_meta.h>

namespace metaspore {

class SparseTensorPartition {
  public:
    SparseTensorMeta &GetMeta() { return meta_; }
    const SparseTensorMeta &GetMeta() const { return meta_; }
    void SetMeta(SparseTensorMeta value) { meta_ = std::move(value); }

    int GetPartitionIndex() const { return partition_index_; }
    void SetPartitionIndex(int value) { partition_index_ = value; }

    void AllocateHashMap();
    void Clear();
    void HandlePush(SmartArray<uint8_t> keys, SmartArray<uint8_t> in, bool is_value);
    SmartArray<uint8_t> HandlePull(SmartArray<uint8_t> keys, bool read_only, bool nan_fill);
    void HandlePushPartition(SmartArray<uint8_t> keys, SmartArray<uint8_t> in, bool data_only,
                             bool skip_existing);
    SmartArray<uint8_t> HandlePullPartition(bool data_only, int index, int count,
                                            SmartArray<uint8_t> &keys);
    void HandlePushMeta(const SparseTensorMeta &meta);
    const SparseTensorMeta &HandlePullMeta();
    void Load(const std::string &dir_path);
    void Save(const std::string &dir_path, bool text_mode);
    void Export(const std::string &dir_path);
    void PruneSmall(double epsilon);
    void PruneOld(int max_age);

  private:
    template <typename T> void DoPruneSmall(double epsilon);

    void TransformIndices(SmartArray<uint8_t> keys, bool pull, bool read_only);
    std::string GetSparsePath(const std::string &dir_path) const;
    std::string GetSparseExportPath(const std::string &dir_path) const;

    static constexpr uint64_t kPaddingKey = 0;
    static constexpr uint64_t kPaddingIndex = uint64_t(-2);
    static constexpr uint64_t kNotFoundIndex = uint64_t(-1);
    SparseTensorMeta meta_;
    int partition_index_ = -1;
    ArrayHashMap<uint64_t, uint8_t> data_;
};

} // namespace metaspore
