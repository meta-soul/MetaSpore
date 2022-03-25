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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <metaspore/actor_process.h>
#include <metaspore/network_utils.h>
#include <metaspore/ps_agent.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/thread_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <stdlib.h>
#include <thread>
#include <time.h>

namespace metaspore {

ActorProcess::ActorProcess(std::shared_ptr<ActorConfig> config)
    : config_(std::move(config)), transport_(MessageTransport::Create(config_)),
      manager_(std::make_unique<NodeManager>(config_)) {
    PSAgentCreator agentCreator = config_->GetAgentCreator();
    if (!agentCreator) {
        std::string serr;
        serr.append(config_->GetThisNodeInfo().ToShortString());
        serr.append(": No agent creator specified.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    agent_ = agentCreator();
    if (!agent_) {
        std::string serr;
        serr.append(config_->GetThisNodeInfo().ToShortString());
        serr.append(": No agent instance returned.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    const NodeRole role = config_->GetNodeRole();
    agent_->actor_process_ = this;
    agent_->is_coordinator_ = (role == NodeRole::Coordinator);
    agent_->is_server_ = (role == NodeRole::Server);
    agent_->is_worker_ = (role == NodeRole::Worker);
    agent_->server_count_ = config_->GetServerCount();
    agent_->worker_count_ = config_->GetWorkerCount();
}

void ActorProcess::Barrier(int group) { manager_->Barrier(group, *this); }

int64_t ActorProcess::Send(const Message &msg) {
    try {
        const int64_t n = transport_->SendMessage(msg);
        send_bytes_ += n;
        if (config_->IsMessageDumpingEnabled())
            spdlog::info("SEND {}", msg.ToString());
        return n;
    } catch (const std::exception &e) {
        std::string serr;
        serr.append(config_->GetThisNodeInfo().ToShortString());
        serr.append(": Fail to send the message with id ");
        serr.append(std::to_string(msg.GetMessageMeta().GetMessageId()));
        serr.append(" to node ");
        serr.append(NodeIdToString(msg.GetMessageMeta().GetReceiver()));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        serr.append("\n\nRoot cause: ");
        serr.append(e.what());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
}

void ActorProcess::Receiving() {
    spdlog::info("ActorProcess::Receiving: {} {}",
                 NodeRoleToString(config_->GetThisNodeInfo().GetRole()), GetThreadIdentifier());
    const int drop_rate = config_->GetDropRate();
    unsigned seed = time(nullptr) + config_->GetThisNodeInfo().GetNodeId();
    for (;;) {
        Message msg;
        try {
            const int64_t n = transport_->ReceiveMessage(msg);
            if (drop_rate > 0 && IsReady()) {
                if (rand_r(&seed) % 100 < drop_rate) {
                    spdlog::debug("DROP {}", msg.ToString());
                    continue;
                }
            }
            receive_bytes_ += n;
        } catch (const std::exception &e) {
            std::string serr;
            serr.append(config_->GetThisNodeInfo().ToShortString());
            serr.append(": Fail to receive message.\n\n");
            serr.append(GetStackTrace());
            serr.append("\n\nRoot cause: ");
            serr.append(e.what());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        if (config_->IsMessageDumpingEnabled())
            spdlog::info("RECV {}", msg.ToString());
        if (msg.GetMessageMeta().GetNodeControl().IsEmpty())
            HandleDataMessage(std::move(msg));
        else {
            const MessageMeta &meta = msg.GetMessageMeta();
            const NodeControlCommand command = meta.GetNodeControl().GetCommand();
            switch (command) {
#undef METASPORE_NODE_CONTROL_COMMAND_DEF
#define METASPORE_NODE_CONTROL_COMMAND_DEF(n)                                                      \
    case NodeControlCommand::n:                                                                    \
        if (Handle##n##Message(msg))                                                               \
            return;                                                                                \
        break;                                                                                     \
        /**/
                METASPORE_NODE_CONTROL_COMMANDS(METASPORE_NODE_CONTROL_COMMAND_DEF)
            default:
                spdlog::warn("Drop unknown message: {}", msg.ToString());
                break;
            }
        }
    }
}

bool ActorProcess::IsReady() {
    std::lock_guard<std::mutex> lock(ready_mutex_);
    return ready_;
}

void ActorProcess::SetIsReady(bool value) {
    std::unique_lock<std::mutex> lock(ready_mutex_);
    ready_ = value;
    if (value)
        ready_cv_.notify_one();
}

void ActorProcess::WaitReady() {
    std::unique_lock<std::mutex> lock(ready_mutex_);
    int elapsed = 0;
    const int timeout = 60 * 1000;
    while (!ready_ && elapsed < timeout) {
        // Sleep until all peers are connected to each other.
        ready_cv_.wait_for(lock, std::chrono::milliseconds(100));
        elapsed += 100;
    }
    if (!ready_) {
        std::string serr;
        serr.append(config_->GetThisNodeInfo().ToShortString());
        serr.append(": Fail to connect to others after waiting for ");
        serr.append(std::to_string(timeout / 1000));
        serr.append(" seconds.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        exit(EXIT_FAILURE);
    }
}

bool ActorProcess::HandleDataMessage(Message &&msg) {
    auto message = std::make_shared<Message>(std::move(msg));
    agent_->HandleMessage(message);
    return false;
}

bool ActorProcess::HandleTerminateMessage(const Message &msg) {
    spdlog::info("{} has stopped.", config_->GetThisNodeInfo().ToShortString());
    return true;
}

bool ActorProcess::HandleAddNodeMessage(const Message &msg) {
    UpdateLocalId(msg);
    if (config_->IsCoordinator())
        CoordinatorHandleAddNode(msg);
    else {
        const NodeControl &control = msg.GetMessageMeta().GetNodeControl();
        for (const NodeInfo &node : control.GetNodes()) {
            std::string addr = node.GetAddress();
            if (!connected_nodes_.count(addr)) {
                transport_->Connect(node);
                connected_nodes_[addr] = node.GetNodeId();
            }
            if (node.GetRole() == NodeRole::Server)
                num_servers_++;
            else if (node.GetRole() == NodeRole::Worker)
                num_workers_++;
        }
        spdlog::info("{} has connected to others.", config_->GetThisNodeInfo().ToShortString());
        SetIsReady(true);
    }
    return false;
}

bool ActorProcess::HandleBarrierMessage(const Message &msg) {
    const NodeControl &control = msg.GetMessageMeta().GetNodeControl();
    if (!msg.GetMessageMeta().IsRequest())
        manager_->NotifyBarrierDone(msg);
    else {
        if (barrier_counter_.empty())
            barrier_counter_.resize(8, 0);
        const int group = control.GetBarrierGroup();
        barrier_counter_.at(group)++;
        spdlog::debug("{}: Barrier counter for node group: {}.",
                      config_->GetThisNodeInfo().ToShortString(), barrier_counter_.at(group));
        const std::vector<int> &nodeIds = manager_->GetNodeIds(group);
        if (barrier_counter_.at(group) == nodeIds.size()) {
            barrier_counter_.at(group) = 0;
            Message res;
            res.GetMessageMeta().SetIsRequest(false);
            res.GetMessageMeta().GetNodeControl().SetCommand(NodeControlCommand::Barrier);
            for (int nodeId : nodeIds) {
                if (!shared_node_mapping_.count(nodeId)) {
                    res.GetMessageMeta().SetReceiver(nodeId);
                    res.GetMessageMeta().SetMessageId(GetMessageId());
                    const int rc = Send(res);
                    if (rc <= 0) {
                        std::string serr;
                        serr.append(config_->GetThisNodeInfo().ToShortString());
                        serr.append(": Fail to send barrier done message, rc: ");
                        serr.append(std::to_string(rc));
                        serr.append(".\n\n");
                        serr.append(GetStackTrace());
                        spdlog::error(serr);
                        throw std::runtime_error(serr);
                    }
                }
            }
        }
    }
    return false;
}

std::unordered_set<int> ActorProcess::GetDeadNodes() {
    std::vector<int> dead;
    std::unordered_set<int> deadSet(dead.begin(), dead.end());
    return deadSet;
}

void ActorProcess::UpdateLocalId(const Message &msg) {
    const size_t numNodes = config_->GetServerCount() + config_->GetWorkerCount();
    const NodeControl &control = msg.GetMessageMeta().GetNodeControl();
    if (msg.GetMessageMeta().GetSender() == -1) {
        if (!config_->IsCoordinator()) {
            std::string serr;
            serr.append(config_->GetThisNodeInfo().ToShortString());
            serr.append(": Only coordinator can receive message from unknown sender.\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        if (control.GetNodes().size() != 1) {
            std::string serr;
            serr.append(config_->GetThisNodeInfo().ToShortString());
            serr.append(
                ": Message from unknown sender must contain exactly one node info, but found ");
            serr.append(std::to_string(control.GetNodes().size()));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        if (nodes_.GetNodeControl().GetNodes().size() < numNodes)
            nodes_.GetNodeControl().AddNode(control.GetNodes().at(0));
        else {
            std::string serr;
            serr.append("Unexpected AddNode message.\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }
    NodeInfo &thisNode = config_->GetThisNodeInfo();
    for (size_t i = 0; i < control.GetNodes().size(); i++) {
        const NodeInfo &node = control.GetNodes().at(i);
        if (node.GetHostName() == thisNode.GetHostName() && node.GetPort() == thisNode.GetPort()) {
            if (thisNode.GetNodeId() == -1)
                thisNode = node;
        }
    }
}

void ActorProcess::CoordinatorHandleAddNode(const Message &msg) {
    recovery_nodes_.GetNodeControl().SetCommand(NodeControlCommand::AddNode);
    const time_t t = time(nullptr);
    const size_t numNodes = config_->GetServerCount() + config_->GetWorkerCount();
    if (nodes_.GetNodeControl().GetNodes().size() == numNodes) {
        std::sort(
            nodes_.GetNodeControl().GetNodes().begin(), nodes_.GetNodeControl().GetNodes().end(),
            [](const NodeInfo &x, const NodeInfo &y) {
                return (x.GetHostName().compare(y.GetHostName()) | (x.GetPort() < y.GetPort())) > 0;
            });
        for (NodeInfo &node : nodes_.GetNodeControl().GetNodes()) {
            const int id = node.GetRole() == NodeRole::Server ? ServerRankToNodeId(num_servers_)
                                                              : WorkerRankToNodeId(num_workers_);
            std::string addr = node.GetAddress();
            if (!connected_nodes_.count(addr)) {
                if (node.GetNodeId() != -1) {
                    std::string serr;
                    serr.append(config_->GetThisNodeInfo().ToShortString());
                    serr.append(
                        ": Node id is expected to be unsigned, but it has been assigned to ");
                    serr.append(std::to_string(node.GetNodeId()));
                    serr.append(".\n\n");
                    serr.append(GetStackTrace());
                    spdlog::error(serr);
                    throw std::runtime_error(serr);
                }
                spdlog::debug("{}: Assign id {} to node {}.",
                              config_->GetThisNodeInfo().ToShortString(), id, node.ToString());
                node.SetNodeId(id);
                transport_->Connect(node);
                manager_->UpdateHeartbeat(node.GetNodeId(), t);
                connected_nodes_[addr] = id;
            } else {
                const int nodeId = connected_nodes_[addr];
                shared_node_mapping_[id] = nodeId;
                node.SetNodeId(nodeId);
            }
            if (node.GetRole() == NodeRole::Server)
                num_servers_++;
            if (node.GetRole() == NodeRole::Worker)
                num_workers_++;
        }
        nodes_.GetNodeControl().AddNode(config_->GetThisNodeInfo());
        nodes_.GetNodeControl().SetCommand(NodeControlCommand::AddNode);
        Message res;
        res.SetMessageMeta(nodes_);
        const std::vector<int> &nodeIds = manager_->GetNodeIds(ServerGroup | WorkerGroup);
        for (int id : nodeIds) {
            if (!shared_node_mapping_.count(id)) {
                res.GetMessageMeta().SetReceiver(id);
                res.GetMessageMeta().SetMessageId(GetMessageId());
                Send(res);
            }
        }
        spdlog::info("{}: The coordinator has connected to {} servers and {} workers.",
                     config_->GetThisNodeInfo().ToShortString(), num_servers_, num_workers_);
        SetIsReady(true);
    } else if (!recovery_nodes_.GetNodeControl().GetNodes().empty()) {
        std::unordered_set<int> deadSet = GetDeadNodes();
        if (recovery_nodes_.GetNodeControl().GetNodes().size() != 1) {
            std::string serr;
            serr.append(config_->GetThisNodeInfo().ToShortString());
            serr.append(": Number of recovery nodes is expected to be one, but found ");
            serr.append(std::to_string(recovery_nodes_.GetNodeControl().GetNodes().size()));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        const NodeInfo &recovery_node = recovery_nodes_.GetNodeControl().GetNodes().at(0);
        transport_->Connect(recovery_node);
        manager_->UpdateHeartbeat(recovery_node.GetNodeId(), t);
        Message res;
        const std::vector<int> &nodeIds = manager_->GetNodeIds(ServerGroup | WorkerGroup);
        for (int id : nodeIds) {
            if (id == recovery_node.GetNodeId() || !deadSet.count(id)) {
                if (id == recovery_node.GetNodeId())
                    res.SetMessageMeta(nodes_);
                else
                    res.SetMessageMeta(recovery_nodes_);
                res.GetMessageMeta().SetReceiver(id);
                res.GetMessageMeta().SetMessageId(GetMessageId());
                Send(res);
            }
        }
    }
}

void ActorProcess::Start() {
    transport_->Start();
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        if (init_stage_ == 0) {
            coordinator_.SetRole(NodeRole::Coordinator);
            coordinator_.SetNodeId(CoordinatorNodeId);
            coordinator_.SetHostName(config_->GetRootUri());
            coordinator_.SetPort(config_->GetRootPort());
            const NodeRole role = config_->GetNodeRole();
            if (role == NodeRole::Coordinator)
                config_->SetThisNodeInfo(coordinator_);
            else {
                std::string ip = config_->GetNodeUri();
                if (ip.empty()) {
                    std::string itf = config_->GetNodeInterface();
                    if (itf.empty())
                        ip = network_utils::get_interface_and_ip(itf);
                    else
                        ip = network_utils::get_ip(itf);
                    if (itf.empty()) {
                        std::string serr = "Fail to get the interface.\n\n";
                        serr.append(GetStackTrace());
                        spdlog::error(serr);
                        throw std::runtime_error(serr);
                    }
                    if (ip.empty()) {
                        std::string serr = "Fail to get ip for interface: " + itf + ".\n\n";
                        serr.append(GetStackTrace());
                        spdlog::error(serr);
                        throw std::runtime_error(serr);
                    }
                }
                int port = config_->GetNodePort();
                if (port == 0)
                    port = network_utils::get_available_port();
                if (port == 0) {
                    std::string serr = "Fail to get an available port. ip: " + ip + "\n\n";
                    serr.append(GetStackTrace());
                    spdlog::error(serr);
                    throw std::runtime_error(serr);
                }
                config_->GetThisNodeInfo().SetRole(role);
                config_->GetThisNodeInfo().SetNodeId(-1);
                config_->GetThisNodeInfo().SetHostName(std::move(ip));
                config_->GetThisNodeInfo().SetPort(port);
            }
            const int retry = (role == NodeRole::Coordinator) ? 0 : config_->GetBindRetry();
            try {
                const int port = transport_->Bind(config_->GetThisNodeInfo(), retry);
                config_->GetThisNodeInfo().SetPort(port);
            } catch (const std::exception &e) {
                std::string serr;
                serr.append(config_->GetThisNodeInfo().ToShortString());
                serr.append(": Fail to bind node.\n\n");
                serr.append(GetStackTrace());
                serr.append("\n\nRoot cause: ");
                serr.append(e.what());
                spdlog::error(serr);
                throw std::runtime_error(serr);
            }
            transport_->Connect(coordinator_);
            std::packaged_task<void()> task([this] { this->Receiving(); });
            receiver_exit_ = std::make_unique<std::future<void>>(task.get_future());
            std::thread receiver_thread(std::move(task));
            receiver_thread.detach();
            init_stage_++;
        }
    }
    if (!config_->IsCoordinator()) {
        NodeInfo node = config_->GetThisNodeInfo();
        Message msg;
        msg.GetMessageMeta().SetReceiver(CoordinatorNodeId);
        msg.GetMessageMeta().GetNodeControl().SetCommand(NodeControlCommand::AddNode);
        msg.GetMessageMeta().GetNodeControl().AddNode(node);
        msg.GetMessageMeta().SetMessageId(GetMessageId());
        Send(msg);
    }
    WaitReady();
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        if (init_stage_ == 1) {
            init_stage_++;
        }
    }
}

void ActorProcess::Run() {
    Start();
    auto cb = config_->GetAgentReadyCallback();
    if (cb)
        cb(agent_);
    if (config_->IsCoordinator()) {
        struct ShutdownGuard {
            std::shared_ptr<PSAgent> agent_;

            ~ShutdownGuard() { agent_->Shutdown(); }
        };
        ShutdownGuard guard{agent_};
        agent_->Run();
    }
    Stop();
}

void ActorProcess::Stop() {
    if (receiver_exit_)
        receiver_exit_->get();
    agent_->Finalize();
    agent_->actor_process_ = nullptr;
    agent_.reset();
    init_stage_ = 0;
    SetIsReady(false);
    connected_nodes_.clear();
    shared_node_mapping_.clear();
    send_bytes_ = 0;
    receive_bytes_ = 0;
    message_counter_ = 0;
    config_->GetThisNodeInfo().SetNodeId(-1);
    barrier_counter_.clear();
    num_servers_ = 0;
    num_workers_ = 0;
    transport_->Stop();
}

} // namespace metaspore
