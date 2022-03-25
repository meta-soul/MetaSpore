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

#include <iostream>
#include <metaspore/actor_process.h>
#include <metaspore/debug.h>
#include <metaspore/ps_agent.h>
#include <metaspore/stack_trace_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

void PSAgent::HandleRequest(PSMessage req) {
    PSMessage res = std::make_shared<Message>();
    SendResponse(req, res);
}

int PSAgent::GetAgentRank() const {
    if (!actor_process_)
        return -1;
    std::shared_ptr<ActorConfig> config = actor_process_->GetConfig();
    const int nodeId = config->GetThisNodeInfo().GetNodeId();
    return NodeIdToRank(nodeId);
}

void PSAgent::Barrier(int group) { actor_process_->Barrier(group); }

void PSAgent::Shutdown() {
    const int group = CoordinatorGroup | ServerGroup | WorkerGroup;
    const std::vector<int> &nodeIds = actor_process_->manager_->GetNodeIds(group);
    for (int nodeId : nodeIds) {
        Message msg;
        msg.GetMessageMeta().SetReceiver(nodeId);
        msg.GetMessageMeta().SetMessageId(actor_process_->GetMessageId());
        msg.GetMessageMeta().GetNodeControl().SetCommand(NodeControlCommand::Terminate);
        actor_process_->transport_->SendMessage(msg);
    }
}

void PSAgent::SendRequest(PSMessage req, SingleCallback cb) {
    req->GetMessageMeta().SetIsRequest(true);
    const int receiverId = req->GetMessageMeta().GetReceiver();
    const std::vector<int> &nodeIds = actor_process_->manager_->GetNodeIds(receiverId);
    if (nodeIds.size() != 1) {
        std::string serr;
        serr.append("Expect one node, but receiverId ");
        serr.append(std::to_string(receiverId));
        serr.append(" specifies ");
        serr.append(std::to_string(nodeIds.size()));
        serr.append(" nodes.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    TrackerEntry entry;
    entry.total = nodeIds.size();
    PSMessage response;
    {
        std::unique_lock<std::mutex> lock(tracker_mutex_);
        const int64_t message_id = actor_process_->GetMessageId();
        req->GetMessageMeta().SetMessageId(message_id);
        tracker_.insert(std::make_pair(message_id, entry));
        actor_process_->Send(*req);
        tracker_cv_.wait(lock, [this, message_id] {
            const TrackerEntry &ent = tracker_.at(message_id);
            return ent.responses.size() == ent.total;
        });
        TrackerEntry &e = tracker_.at(message_id);
        std::vector<PSMessage> responses = std::move(e.responses);
        response = responses.at(0);
        e.Clear();
        tracker_.erase(message_id);
    }
    if (response->GetMessageMeta().IsException()) {
        std::string serr;
        serr.append(NodeIdToString(response->GetMessageMeta().GetReceiver()));
        serr.append(": remote node ");
        serr.append(NodeIdToString(response->GetMessageMeta().GetSender()));
        serr.append(" returned exception. ");
        serr.append(response->GetMessageMeta().GetBody());
        serr.append("\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    cb(req, response);
}

void PSAgent::SendAllRequests(std::vector<PSMessage> reqs, MultipleCallback cb) {
    for (PSMessage &req : reqs) {
        const int receiverId = req->GetMessageMeta().GetReceiver();
        const std::vector<int> &nodeIds = actor_process_->manager_->GetNodeIds(receiverId);
        if (nodeIds.size() != 1) {
            std::string serr;
            serr.append("Expect one node, but receiverId ");
            serr.append(std::to_string(receiverId));
            serr.append(" specifies ");
            serr.append(std::to_string(nodeIds.size()));
            serr.append(" nodes.\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }
    TrackerEntry entry;
    entry.total = reqs.size();
    std::vector<PSMessage> responses;
    {
        std::unique_lock<std::mutex> lock(tracker_mutex_);
        const int64_t message_id = actor_process_->GetMessageId();
        tracker_.insert(std::make_pair(message_id, entry));
        for (const PSMessage &req : reqs) {
            req->GetMessageMeta().SetIsRequest(true);
            req->GetMessageMeta().SetMessageId(message_id);
            actor_process_->Send(*req);
        }
        tracker_cv_.wait(lock, [this, message_id] {
            const TrackerEntry &ent = tracker_.at(message_id);
            return ent.responses.size() == ent.total;
        });
        TrackerEntry &e = tracker_.at(message_id);
        responses = std::move(e.responses);
        e.Clear();
        tracker_.erase(message_id);
    }
    for (PSMessage &res : responses) {
        if (res->GetMessageMeta().IsException()) {
            std::string serr;
            serr.append(NodeIdToString(res->GetMessageMeta().GetReceiver()));
            serr.append(": remote node ");
            serr.append(NodeIdToString(res->GetMessageMeta().GetSender()));
            serr.append(" returned exception. ");
            serr.append(res->GetMessageMeta().GetBody());
            serr.append("\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }
    cb(std::move(reqs), std::move(responses));
}

void PSAgent::BroadcastRequest(PSMessage req, BroadcastCallback cb) {
    req->GetMessageMeta().SetIsRequest(true);
    const int receiverId = req->GetMessageMeta().GetReceiver();
    const std::vector<int> &nodeIds = actor_process_->manager_->GetNodeIds(receiverId);
    if (nodeIds.empty()) {
        std::string serr;
        serr.append("Expect one or more nodes, but receiverId ");
        serr.append(std::to_string(receiverId));
        serr.append(" specifies ");
        serr.append(std::to_string(nodeIds.size()));
        serr.append(" nodes.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    TrackerEntry entry;
    entry.total = nodeIds.size();
    std::vector<PSMessage> responses;
    {
        std::unique_lock<std::mutex> lock(tracker_mutex_);
        const int64_t message_id = actor_process_->GetMessageId();
        req->GetMessageMeta().SetMessageId(message_id);
        tracker_.insert(std::make_pair(message_id, entry));
        for (int nodeId : nodeIds) {
            req->GetMessageMeta().SetReceiver(nodeId);
            actor_process_->Send(*req);
        }
        tracker_cv_.wait(lock, [this, message_id] {
            const TrackerEntry &ent = tracker_.at(message_id);
            return ent.responses.size() == ent.total;
        });
        TrackerEntry &e = tracker_.at(message_id);
        responses = std::move(e.responses);
        e.Clear();
        tracker_.erase(message_id);
    }
    for (PSMessage &res : responses) {
        if (res->GetMessageMeta().IsException()) {
            std::string serr;
            serr.append(NodeIdToString(res->GetMessageMeta().GetReceiver()));
            serr.append(": remote node ");
            serr.append(NodeIdToString(res->GetMessageMeta().GetSender()));
            serr.append(" returned exception. ");
            serr.append(res->GetMessageMeta().GetBody());
            serr.append("\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }
    cb(req, std::move(responses));
}

void PSAgent::SendResponse(PSMessage req, PSMessage res) {
    try {
        const int sender = req->GetMessageMeta().GetSender();
        const int64_t message_id = req->GetMessageMeta().GetMessageId();
        res->GetMessageMeta().SetReceiver(sender);
        res->GetMessageMeta().SetMessageId(message_id);
        res->GetMessageMeta().SetIsRequest(false);
        actor_process_->Send(*res);
    } catch (const std::exception &e) {
        std::string serr;
        serr.append("Fail to send response from ");
        serr.append(NodeIdToString(req->GetMessageMeta().GetReceiver()));
        serr.append(" back to ");
        serr.append(NodeIdToString(req->GetMessageMeta().GetSender()));
        serr.append(".\n\nreq:\n");
        serr.append(req->ToString());
        serr.append("\n\nres:\n");
        serr.append(res->ToString());
        serr.append("\n\n");
        serr.append(GetStackTrace());
        serr.append("\n\nRoot cause: ");
        serr.append(e.what());
        throw std::runtime_error(serr);
    }
}

void PSAgent::HandleMessage(PSMessage msg) {
    if (msg->GetMessageMeta().IsRequest()) {
        try {
            HandleRequest(msg);
        } catch (const std::exception &e) {
            PSMessage exc = std::make_shared<Message>();
            exc->GetMessageMeta().SetIsException(true);
            exc->GetMessageMeta().SetBody(e.what());
            SendResponse(msg, exc);
        }
    } else {
        const int64_t message_id = msg->GetMessageMeta().GetMessageId();
        std::lock_guard<std::mutex> lock(tracker_mutex_);
        TrackerEntry &ent = tracker_.at(message_id);
        ent.responses.push_back(msg);
        if (ent.responses.size() == ent.total)
            tracker_cv_.notify_all();
    }
}

std::string PSAgent::ToString() const {
    if (!actor_process_)
        return "?";
    std::shared_ptr<ActorConfig> config = actor_process_->GetConfig();
    std::string str = config->GetThisNodeInfo().ToShortString();
    return str;
}

} // namespace metaspore
