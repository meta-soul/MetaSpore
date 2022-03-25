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
#include <functional>
#include <memory>
#include <metaspore/ps_agent.h>
#include <metaspore/smart_array.h>
#include <metaspore/sparse_tensor_meta.h>

namespace metaspore {

class SparseTensor {
  public:
    SparseTensorMeta &GetMeta() { return meta_; }
    const SparseTensorMeta &GetMeta() const { return meta_; }
    void SetMeta(SparseTensorMeta value) { meta_ = std::move(value); }

    std::shared_ptr<PSAgent> GetAgent() const { return agent_; }
    void SetAgent(std::shared_ptr<PSAgent> value) { agent_ = std::move(value); }

    void Init(std::function<void()> cb);
    void Dispose(std::function<void()> cb);
    void Clear(std::function<void()> cb);
    void Push(SmartArray<uint8_t> keys, SmartArray<uint8_t> in, std::function<void()> cb,
              bool is_value = false);
    void Pull(SmartArray<uint8_t> keys, std::function<void(SmartArray<uint8_t> out)> cb,
              bool read_only = false, bool nan_fill = false);
    void PushPartition(ArrayHashMap<uint64_t, uint8_t> &data, std::function<void()> cb,
                       bool data_only = false, bool skip_existing = false);
    void PullPartition(ArrayHashMap<uint64_t, uint8_t> &data, std::function<void()> cb,
                       bool data_only = false, int index = -1, int count = -1);
    void PushMeta(const SparseTensorMeta &meta, std::function<void()> cb);
    void PullMeta(std::function<void(SparseTensorMeta meta)> cb);
    void Load(const std::string &dir_path, std::function<void()> cb, bool keep_meta = false);
    void Save(const std::string &dir_path, std::function<void()> cb, bool text_mode = false);
    void Export(const std::string &dir_path, std::function<void()> cb);
    void ImportFrom(const std::string &meta_file_path, std::function<void()> cb,
                    bool data_only = false, bool skip_existing = false, bool transform_key = false,
                    const std::string &feature_name = "");
    void PruneSmall(double epsilon, std::function<void()> cb);
    void PruneOld(int max_age, std::function<void()> cb);

  private:
    std::string GetSparseMetaPath(const std::string &dir_path) const;
    static std::string GetSparsePath(const std::string &dir_path, const SparseTensorMeta &meta,
                                     int index);

    SparseTensorMeta meta_;
    std::shared_ptr<PSAgent> agent_;
};

} // namespace metaspore
