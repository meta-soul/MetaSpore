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

#include <functional>
#include <memory>
#include <metaspore/dense_tensor_meta.h>
#include <metaspore/ps_agent.h>
#include <metaspore/smart_array.h>

namespace metaspore {

class DenseTensor {
  public:
    DenseTensorMeta &GetMeta() { return meta_; }
    const DenseTensorMeta &GetMeta() const { return meta_; }
    void SetMeta(DenseTensorMeta value) { meta_ = std::move(value); }

    std::shared_ptr<PSAgent> GetAgent() const { return agent_; }
    void SetAgent(std::shared_ptr<PSAgent> value) { agent_ = std::move(value); }

    void Init(std::function<void()> cb);
    void Dispose(std::function<void()> cb);
    void Push(SmartArray<uint8_t> in, std::function<void()> cb, bool is_value = false,
              bool is_state = false);
    void Pull(std::function<void(SmartArray<uint8_t> out)> cb, bool is_state = false);
    void PushMeta(const DenseTensorMeta &meta, std::function<void()> cb);
    void PullMeta(std::function<void(DenseTensorMeta meta)> cb);
    void Load(const std::string &dir_path, std::function<void()> cb, bool keep_meta = false);
    void Save(const std::string &dir_path, std::function<void()> cb);

  private:
    std::string GetDenseMetaPath(const std::string &dir_path) const;
    std::string GetDenseDataPath(const std::string &dir_path) const;
    std::string GetDenseStatePath(const std::string &dir_path) const;

    DenseTensorMeta meta_;
    std::shared_ptr<PSAgent> agent_;
};

} // namespace metaspore
