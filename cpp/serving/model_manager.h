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

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include <serving/grpc_model_runner.h>
#include <common/types.h>

namespace metaspore::serving {

class ModelManager {
  public:
    static ModelManager &get_model_manager() {
        static ModelManager mgr;
        return mgr;
    }

    void init(const std::string &dir_path);

    awaitable_status load(const std::string &dir_path, const std::string &name);

    result<std::shared_ptr<GrpcModelRunner>> get_model(const std::string &name);

  private:
    std::unordered_map<std::string, std::shared_ptr<GrpcModelRunner>> models_;
    std::shared_mutex mu_;

    ModelManager() = default;
    ModelManager(const ModelManager &) = delete;
    ModelManager(ModelManager &&) = delete;
    ModelManager &operator=(const ModelManager &) = delete;
    ModelManager &operator=(ModelManager &&) = delete;
};

} // namespace metaspore::serving
