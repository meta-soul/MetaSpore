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

#include <metaspore/actor_process.h>
#include <metaspore/node_manager.h>
#include <metaspore/node_role.h>
#include <metaspore/stack_trace_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

NodeManager::NodeManager(std::shared_ptr<ActorConfig> config) : config_(std::move(config)) {
    InitNodeIds();
}

void NodeManager::InitNodeIds() {
    node_ids_.clear();
    for (int i = 0; i < config_->GetWorkerCount(); i++) {
        const int id = WorkerRankToNodeId(i);
        for (int g : {id, WorkerGroup, WorkerGroup | ServerGroup, WorkerGroup | CoordinatorGroup,
                      WorkerGroup | CoordinatorGroup | ServerGroup})
            node_ids_[g].push_back(id);
    }
    for (int i = 0; i < config_->GetServerCount(); i++) {
        const int id = ServerRankToNodeId(i);
        for (int g : {id, ServerGroup, ServerGroup | WorkerGroup, ServerGroup | CoordinatorGroup,
                      ServerGroup | CoordinatorGroup | WorkerGroup})
            node_ids_[g].push_back(id);
    }
    for (int g : {CoordinatorNodeId, CoordinatorGroup, CoordinatorGroup | ServerGroup,
                  CoordinatorGroup | WorkerGroup, CoordinatorGroup | ServerGroup | WorkerGroup})
        node_ids_[g].push_back(CoordinatorNodeId);
}

const std::vector<int> &NodeManager::GetNodeIds(int group) const {
    auto it = node_ids_.find(group);
    if (it == node_ids_.end()) {
        std::string serr;
        serr.append("Node group ");
        serr.append(std::to_string(group));
        serr.append(" does not exist.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    return it->second;
}

void NodeManager::Barrier(int group, ActorProcess &process) {
    const std::vector<int> &nodeIds = GetNodeIds(group);
    if (nodeIds.size() <= 1)
        return;
    if (config_->IsCoordinator() && ((group & CoordinatorGroup) == 0) ||
        config_->IsServer() && ((group & ServerGroup) == 0) ||
        config_->IsWorker() && ((group & WorkerGroup) == 0)) {
        std::string serr;
        serr.append("Barrier group ");
        serr.append(std::to_string(group));
        serr.append(" does not contain the current node.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    std::unique_lock<std::mutex> lock(barrier_mutex_);
    barrier_done_ = false;
    Message req;
    req.GetMessageMeta().SetReceiver(CoordinatorNodeId);
    req.GetMessageMeta().SetIsRequest(true);
    req.GetMessageMeta().GetNodeControl().SetCommand(NodeControlCommand::Barrier);
    req.GetMessageMeta().GetNodeControl().SetBarrierGroup(group);
    req.GetMessageMeta().SetMessageId(process.GetMessageId());
    int rc = process.Send(req);
    if (rc <= 0) {
        std::string serr;
        serr.append("Fail to send barrier message to node group ");
        serr.append(std::to_string(group));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    barrier_cv_.wait(lock, [this] { return barrier_done_; });
}

void NodeManager::NotifyBarrierDone(const Message &msg) {
    const NodeControl &control = msg.GetMessageMeta().GetNodeControl();
    if (control.GetCommand() == NodeControlCommand::Barrier && !msg.GetMessageMeta().IsRequest()) {
        {
            std::lock_guard<std::mutex> lock(barrier_mutex_);
            barrier_done_ = true;
        }
        barrier_cv_.notify_all();
    }
}

void NodeManager::UpdateHeartbeat(int nodeId, time_t t) {
    std::lock_guard<std::mutex> lock(heartbeat_mutex_);
    heartbeats_[nodeId] = t;
}

std::vector<int> NodeManager::GetDeadNodes(int timeout) {
    std::vector<int> dead;
    const std::vector<int> &nodeIds = config_->IsCoordinator()
                                          ? GetNodeIds(ServerGroup | WorkerGroup)
                                          : GetNodeIds(CoordinatorGroup);
    const time_t now = time(nullptr);
    std::lock_guard<std::mutex> lock(heartbeat_mutex_);
    for (int id : nodeIds) {
        auto it = heartbeats_.find(id);
        if (it == heartbeats_.end() || it->second + timeout < now && start_time_ + timeout < now)
            dead.push_back(id);
    }
    return dead;
}

} // namespace metaspore
