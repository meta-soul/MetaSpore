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

#include <metaspore/dense_tensor_meta.h>

namespace metaspore {

class DenseTensorPartition {
  public:
    DenseTensorMeta &GetMeta() { return meta_; }
    const DenseTensorMeta &GetMeta() const { return meta_; }
    void SetMeta(DenseTensorMeta value) { meta_ = std::move(value); }

    std::vector<size_t> &GetPartitionDataShape() { return partition_data_shape_; }
    const std::vector<size_t> &GetPartitionDataShape() const { return partition_data_shape_; }
    void SetPartitionDataShape(std::vector<size_t> value) {
        partition_data_shape_ = std::move(value);
    }

    std::vector<size_t> &GetPartitionStateShape() { return partition_state_shape_; }
    const std::vector<size_t> &GetPartitionStateShape() const { return partition_state_shape_; }
    void SetPartitionStateShape(std::vector<size_t> value) {
        partition_state_shape_ = std::move(value);
    }

    int GetPartitionIndex() const { return partition_index_; }
    void SetPartitionIndex(int value) { partition_index_ = value; }

    size_t GetOffset() const { return offset_; }
    void SetOffset(size_t value) { offset_ = value; }

    bool IsEmpty() const { return data_.empty(); }

    void AllocateDataBlock(bool init);
    void HandlePush(SmartArray<uint8_t> in, bool is_value, bool is_state);
    SmartArray<uint8_t> HandlePull(bool is_state);
    void HandlePushMeta(const DenseTensorMeta &meta);
    const DenseTensorMeta &HandlePullMeta();

  private:
    DenseTensorMeta meta_;
    std::vector<size_t> partition_data_shape_;
    std::vector<size_t> partition_state_shape_;
    int partition_index_ = -1;
    size_t offset_ = size_t(-1);
    SmartArray<uint8_t> data_;
    SmartArray<uint8_t> state_;
};

} // namespace metaspore
