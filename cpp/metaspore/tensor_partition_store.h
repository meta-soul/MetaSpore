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

#include <metaspore/dense_tensor_partition.h>
#include <metaspore/message.h>
#include <metaspore/ps_agent.h>
#include <metaspore/sparse_tensor_partition.h>
#include <unordered_map>

namespace metaspore {

class TensorPartitionStore {
  public:
    int GetPartitionCount() const { return partition_count_; }
    void SetPartitionCount(int value) { partition_count_ = value; }

    int GetPartitionIndex() const { return partition_index_; }
    void SetPartitionIndex(int value) { partition_index_ = value; }

    void DenseInit(const DenseTensorMeta &meta);
    void DenseDispose(const std::string &name);
    void DensePush(const std::string &name, PSMessage req, bool is_value, bool is_state);
    PSMessage DensePull(const std::string &name, bool is_state);
    void DensePushMeta(const std::string &name, const DenseTensorMeta &meta);
    PSMessage DensePullMeta(const std::string &name);

    void SparseInit(const SparseTensorMeta &meta);
    void SparseDispose(const std::string &name);
    void SparseClear(const std::string &name);
    void SparsePush(const std::string &name, PSMessage req, bool is_value);
    PSMessage SparsePull(const std::string &name, PSMessage req, bool read_only, bool nan_fill);
    void SparsePushPartition(const std::string &name, PSMessage req, bool data_only,
                             bool skip_existing);
    PSMessage SparsePullPartition(const std::string &name, bool data_only, int index, int count);
    void SparsePushMeta(const std::string &name, const SparseTensorMeta &meta);
    PSMessage SparsePullMeta(const std::string &name);
    void SparseLoad(const std::string &name, const std::string &dir_path);
    void SparseSave(const std::string &name, const std::string &dir_path, bool text_mode);
    void SparseExport(const std::string &name, const std::string &dir_path);
    void SparsePruneSmall(const std::string &name, double epsilon);
    void SparsePruneOld(const std::string &name, int max_age);

  private:
    int partition_count_ = -1;
    int partition_index_ = -1;
    std::unordered_map<std::string, DenseTensorPartition> dense_store_;
    std::unordered_map<std::string, SparseTensorPartition> sparse_store_;
};

} // namespace metaspore
