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
#include <metaspore/node_info.h>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace metaspore {

class ActorConfig : public std::enable_shared_from_this<ActorConfig> {
  public:
    using AgentCreator = std::function<std::shared_ptr<class PSAgent>()>;
    using AgentReadyCallback = std::function<void(std::shared_ptr<class PSAgent>)>;

    AgentCreator GetAgentCreator() const { return agent_creator_; }
    void SetAgentCreator(AgentCreator value) { agent_creator_ = std::move(value); }

    AgentReadyCallback GetAgentReadyCallback() const { return agent_ready_callback_; }
    void SetAgentReadyCallback(AgentReadyCallback value) {
        agent_ready_callback_ = std::move(value);
    }

    const std::string &GetTransportType() const { return transport_type_; }
    void SetTransportType(std::string value) { transport_type_ = std::move(value); }

    bool IsLocalMode() const { return is_local_mode_; }
    void SetIsLocalMode(bool value) { is_local_mode_ = value; }

    bool UseKubernetes() const { return use_kubernetes_; }
    void SetUseKubernetes(bool value) { use_kubernetes_ = value; }

    const std::string &GetRootUri() const { return root_uri_; }
    void SetRootUri(std::string value) { root_uri_ = std::move(value); }

    int GetRootPort() const { return root_port_; }
    void SetRootPort(int value) { root_port_ = value; }

    const std::string &GetNodeUri() const { return node_uri_; }
    void SetNodeUri(std::string value) { node_uri_ = std::move(value); }

    const std::string &GetNodeInterface() const { return node_interface_; }
    void SetNodeInterface(std::string value) { node_interface_ = std::move(value); }

    NodeRole GetNodeRole() const { return node_role_; }
    void SetNodeRole(NodeRole value) { node_role_ = value; }

    int GetNodePort() { return node_port_; }
    void SetNodePort(int value) { node_port_ = value; }

    NodeInfo &GetThisNodeInfo() { return this_node_info_; }
    const NodeInfo &GetThisNodeInfo() const { return this_node_info_; }
    void SetThisNodeInfo(NodeInfo value) { this_node_info_ = std::move(value); }

    bool IsCoordinator() const { return this_node_info_.GetRole() == NodeRole::Coordinator; }
    bool IsServer() const { return this_node_info_.GetRole() == NodeRole::Server; }
    bool IsWorker() const { return this_node_info_.GetRole() == NodeRole::Worker; }

    int GetBindRetry() const { return bind_retry_; }
    void SetBindRetry(int value) { bind_retry_ = value; }

    int GetHeartbeatInterval() const { return heartbeat_interval_; }
    void SetHeartbeatInterval(int value) { heartbeat_interval_ = value; }

    int GetHeartbeatTimeout() const { return heartbeat_timeout_; }
    void SetHeartbeatTimeout(int value) { heartbeat_timeout_ = value; }

    bool IsMessageDumpingEnabled() { return is_message_dumping_enabled_; }
    void SetIsMessageDumpingEnabled(bool value) { is_message_dumping_enabled_ = value; }

    bool IsResendingEnabled() const { return is_resending_enabled_; }
    void SetIsResendingEnabled(bool value) { is_resending_enabled_ = value; }

    int GetResendingTimeout() const { return resending_timeout_; }
    void SetResendingTimeout(int value) { resending_timeout_ = value; }

    int GetResendingRetry() const { return resending_retry_; }
    void SetResendingRetry(int value) { resending_retry_ = value; }

    int GetDropRate() const { return drop_rate_; }
    void SetDropRate(int value) { drop_rate_ = value; }

    int GetServerCount() { return server_count_; }
    void SetServerCount(int value) { server_count_ = value; }

    int GetWorkerCount() { return worker_count_; }
    void SetWorkerCount(int value) { worker_count_ = value; }

    std::shared_ptr<ActorConfig> Copy() const { return std::make_shared<ActorConfig>(*this); }

  private:
    static constexpr const char *default_transport_type = "ZeroMQ";
    static constexpr int default_bind_retry = 40;
    static constexpr int default_heartbeat_interval = 0;
    static constexpr int default_heartbeat_timeout = 0;
    static constexpr int default_resending_timeout = 1000;
    static constexpr int default_resending_retry = 10;

    AgentCreator agent_creator_;
    AgentReadyCallback agent_ready_callback_;
    std::string transport_type_ = default_transport_type;
    bool is_local_mode_ = false;
    bool use_kubernetes_ = false;
    std::string root_uri_;
    int root_port_ = 0;
    std::string node_uri_;
    std::string node_interface_;
    int node_port_ = 0;
    NodeRole node_role_;
    NodeInfo this_node_info_;
    int bind_retry_ = default_bind_retry;
    int heartbeat_interval_ = default_heartbeat_interval;
    int heartbeat_timeout_ = default_heartbeat_timeout;
    bool is_message_dumping_enabled_ = false;
    bool is_resending_enabled_ = false;
    int resending_timeout_ = default_resending_timeout;
    int resending_retry_ = default_resending_retry;
    int drop_rate_ = 0;
    int server_count_ = 0;
    int worker_count_ = 0;
};

} // namespace metaspore
