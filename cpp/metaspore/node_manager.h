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

#include <condition_variable>
#include <memory>
#include <metaspore/actor_config.h>
#include <metaspore/message.h>
#include <mutex>
#include <time.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metaspore {

class ActorProcess;

class NodeManager {
  public:
    explicit NodeManager(std::shared_ptr<ActorConfig> config);

    std::shared_ptr<ActorConfig> GetConfig() const { return config_; }
    void SetConfig(std::shared_ptr<ActorConfig> value) { config_ = std::move(value); }

    const std::vector<int> &GetNodeIds(int group) const;
    void Barrier(int group, ActorProcess &process);
    void NotifyBarrierDone(const Message &msg);
    void UpdateHeartbeat(int nodeId, time_t t);
    std::vector<int> GetDeadNodes(int timeout);

  private:
    void InitNodeIds();

    std::shared_ptr<ActorConfig> config_;
    std::mutex start_mutex_;
    time_t start_time_ = 0;
    std::unordered_map<int, std::vector<int>> node_ids_;
    std::mutex barrier_mutex_;
    std::condition_variable barrier_cv_;
    bool barrier_done_;
    std::mutex heartbeat_mutex_;
    std::unordered_map<int, time_t> heartbeats_;
};

} // namespace metaspore
