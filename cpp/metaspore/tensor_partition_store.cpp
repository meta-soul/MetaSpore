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

#include <metaspore/io.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/tensor_partition_store.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

void TensorPartitionStore::DenseInit(const DenseTensorMeta &meta) {
    if (sparse_store_.count(meta.GetName())) {
        std::string serr;
        serr.append("Can not initialize dense tensor '");
        serr.append(meta.GetName());
        serr.append("', as it has been initialized as a sparse tensor.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    auto it = dense_store_.find(meta.GetName());
    if (it == dense_store_.end()) {
        DenseTensorPartition &part = dense_store_[meta.GetName()];
        part.SetMeta(meta);
        part.SetPartitionIndex(partition_index_);
        part.AllocateDataBlock(true);
    } else {
        const DenseTensorMeta &existing = it->second.GetMeta();
        if (meta != existing) {
            std::string serr;
            serr.append("Can not initialize dense tensor '");
            serr.append(meta.GetName());
            serr.append("' with different meta; meta: ");
            serr.append(meta.ToString());
            serr.append(", existing: ");
            serr.append(existing.ToString());
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }
}

void TensorPartitionStore::DenseDispose(const std::string &name) {
    auto it = dense_store_.find(name);
    if (it == dense_store_.end()) {
        std::string serr;
        serr.append("Dense tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    dense_store_.erase(it);
}

void TensorPartitionStore::DensePush(const std::string &name, PSMessage req, bool is_value,
                                     bool is_state) {
    auto it = dense_store_.find(name);
    if (it == dense_store_.end()) {
        std::string serr;
        serr.append("Dense tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    DenseTensorPartition &part = it->second;
    SmartArray<uint8_t> in = req->GetTypedSlice(0, part.GetMeta().GetDataType());
    part.HandlePush(in, is_value, is_state);
}

PSMessage TensorPartitionStore::DensePull(const std::string &name, bool is_state) {
    auto it = dense_store_.find(name);
    if (it == dense_store_.end()) {
        std::string serr;
        serr.append("Dense tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    DenseTensorPartition &part = it->second;
    SmartArray<uint8_t> out = part.HandlePull(is_state);
    PSMessage res = std::make_shared<Message>();
    res->AddTypedSlice(out, part.GetMeta().GetDataType());
    return res;
}

void TensorPartitionStore::DensePushMeta(const std::string &name, const DenseTensorMeta &meta) {
    auto it = dense_store_.find(name);
    if (it == dense_store_.end()) {
        std::string serr;
        serr.append("Dense tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    DenseTensorPartition &part = it->second;
    part.HandlePushMeta(meta);
}

PSMessage TensorPartitionStore::DensePullMeta(const std::string &name) {
    auto it = dense_store_.find(name);
    if (it == dense_store_.end()) {
        std::string serr;
        serr.append("Dense tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    DenseTensorPartition &part = it->second;
    PSMessage res = std::make_shared<Message>();
    if (!part.IsEmpty()) {
        const DenseTensorMeta &meta = part.HandlePullMeta();
        res->GetMessageMeta().SetBody(meta.ToJsonString());
    }
    return res;
}

void TensorPartitionStore::SparseInit(const SparseTensorMeta &meta) {
    if (dense_store_.count(meta.GetName())) {
        std::string serr;
        serr.append("Can not initialize sparse tensor '");
        serr.append(meta.GetName());
        serr.append("', as it has been initialized as a dense tensor.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    auto it = sparse_store_.find(meta.GetName());
    if (it == sparse_store_.end()) {
        SparseTensorPartition &part = sparse_store_[meta.GetName()];
        part.SetMeta(meta);
        part.SetPartitionIndex(partition_index_);
        part.AllocateHashMap();
    } else {
        const SparseTensorMeta &existing = it->second.GetMeta();
        if (meta != existing) {
            std::string serr;
            serr.append("Can not initialize sparse tensor '");
            serr.append(meta.GetName());
            serr.append("' with different meta; meta: ");
            serr.append(meta.ToString());
            serr.append(", existing: ");
            serr.append(existing.ToString());
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }
}

void TensorPartitionStore::SparseDispose(const std::string &name) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    sparse_store_.erase(it);
}

void TensorPartitionStore::SparseClear(const std::string &name) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    part.Clear();
}

void TensorPartitionStore::SparsePush(const std::string &name, PSMessage req, bool is_value) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    SmartArray<uint8_t> keys = req->GetTypedSlice<uint64_t>(0).Cast<uint8_t>();
    SmartArray<uint8_t> in = req->GetTypedSlice(1, part.GetMeta().GetDataType());
    part.HandlePush(keys, in, is_value);
}

PSMessage TensorPartitionStore::SparsePull(const std::string &name, PSMessage req, bool read_only,
                                           bool nan_fill) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    SmartArray<uint8_t> keys = req->GetTypedSlice<uint64_t>(0).Cast<uint8_t>();
    SmartArray<uint8_t> out = part.HandlePull(keys, read_only, nan_fill);
    PSMessage res = std::make_shared<Message>();
    res->AddTypedSlice(out, part.GetMeta().GetDataType());
    return res;
}

void TensorPartitionStore::SparsePushPartition(const std::string &name, PSMessage req,
                                               bool data_only, bool skip_existing) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    SmartArray<uint8_t> keys = req->GetTypedSlice<uint64_t>(0).Cast<uint8_t>();
    SmartArray<uint8_t> in = req->GetTypedSlice(1, part.GetMeta().GetDataType());
    part.HandlePushPartition(keys, in, data_only, skip_existing);
}

PSMessage TensorPartitionStore::SparsePullPartition(const std::string &name, bool data_only,
                                                    int index, int count) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    SmartArray<uint8_t> keys;
    SmartArray<uint8_t> out = part.HandlePullPartition(data_only, index, count, keys);
    PSMessage res = std::make_shared<Message>();
    res->AddTypedSlice(keys, DataType::UInt64);
    res->AddTypedSlice(out, part.GetMeta().GetDataType());
    return res;
}

void TensorPartitionStore::SparsePushMeta(const std::string &name, const SparseTensorMeta &meta) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    part.HandlePushMeta(meta);
}

PSMessage TensorPartitionStore::SparsePullMeta(const std::string &name) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    PSMessage res = std::make_shared<Message>();
    if (!part.IsEmpty()) {
        const SparseTensorMeta &meta = part.HandlePullMeta();
        res->GetMessageMeta().SetBody(meta.ToJsonString());
    }
    return res;
}

void TensorPartitionStore::SparseLoad(const std::string &name, const std::string &dir_path) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    part.Load(dir_path);
}

void TensorPartitionStore::SparseSave(const std::string &name, const std::string &dir_path,
                                      bool text_mode) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    EnsureLocalDirectory(dir_path);
    part.Save(dir_path, text_mode);
}

void TensorPartitionStore::SparseExport(const std::string &name, const std::string &dir_path) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    EnsureLocalDirectory(dir_path);
    part.Export(dir_path);
}

void TensorPartitionStore::SparsePruneSmall(const std::string &name, double epsilon) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    part.PruneSmall(epsilon);
}

void TensorPartitionStore::SparsePruneOld(const std::string &name, int max_age) {
    auto it = sparse_store_.find(name);
    if (it == sparse_store_.end()) {
        std::string serr;
        serr.append("Sparse tensor '");
        serr.append(name);
        serr.append("' does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    SparseTensorPartition &part = it->second;
    part.PruneOld(max_age);
}

} // namespace metaspore
