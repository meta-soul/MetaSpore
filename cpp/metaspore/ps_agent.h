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
#include <functional>
#include <memory>
#include <metaspore/message.h>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metaspore {

using PSMessage = std::shared_ptr<Message>;
using SingleCallback = std::function<void(PSMessage req, PSMessage res)>;
using MultipleCallback =
    std::function<void(std::vector<PSMessage> reqs, std::vector<PSMessage> ress)>;
using BroadcastCallback = std::function<void(PSMessage req, std::vector<PSMessage> ress)>;
using PSAgentCreator = std::function<std::shared_ptr<class PSAgent>()>;

class PSAgent : public std::enable_shared_from_this<PSAgent> {
    friend class ActorProcess;

  public:
    virtual ~PSAgent() {}

    virtual void Run() {}
    virtual void HandleRequest(PSMessage req);
    virtual void Finalize() {}

    bool IsCoordinator() const { return is_coordinator_; }
    bool IsServer() const { return is_server_; }
    bool IsWorker() const { return is_worker_; }

    int GetServerCount() const { return server_count_; }
    int GetWorkerCount() const { return worker_count_; }
    int GetAgentRank() const;

    void Barrier(int group);
    void Shutdown();

    void SendRequest(PSMessage req, SingleCallback cb);
    void SendAllRequests(std::vector<PSMessage> reqs, MultipleCallback cb);
    void BroadcastRequest(PSMessage req, BroadcastCallback cb);
    void SendResponse(PSMessage req, PSMessage res);
    void HandleMessage(PSMessage msg);

    std::string ToString() const;

  private:
    class ActorProcess *actor_process_ = nullptr;

    struct TrackerEntry {
        int total = 0;
        std::vector<PSMessage> responses;

        void Clear() { responses.clear(); }
    };

    std::mutex tracker_mutex_;
    std::condition_variable tracker_cv_;
    std::unordered_map<int64_t, TrackerEntry> tracker_;

    bool is_coordinator_ = false;
    bool is_server_ = false;
    bool is_worker_ = false;

    int server_count_ = 0;
    int worker_count_ = 0;
};

} // namespace metaspore
