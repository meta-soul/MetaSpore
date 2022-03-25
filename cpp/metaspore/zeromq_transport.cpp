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
#include <metaspore/stack_trace_utils.h>
#include <metaspore/zeromq_transport.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zmq.h>

namespace metaspore {

ZeroMQTransport::ZeroMQTransport(std::shared_ptr<ActorConfig> config)
    : MessageTransport(std::move(config)) {}

void ZeroMQTransport::Start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (context_ == nullptr) {
        context_ = zmq_ctx_new();
        if (context_ == nullptr) {
            std::string serr = "Fail to create ZeroMQ context.\n\n";
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        zmq_ctx_set(context_, ZMQ_MAX_SOCKETS, 65536);
    }
}

void ZeroMQTransport::Stop() {
    int linger = 0;
    int rc = zmq_setsockopt(receiver_, ZMQ_LINGER, &linger, sizeof(linger));
    if (rc != 0 && errno != ETERM) {
        std::string serr;
        serr.append("zmq_setsockopt with ZMQ_LINGER is expected to return ");
        serr.append("0 or ETERM, but the actual rc is ");
        serr.append(std::to_string(rc));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    rc = zmq_close(receiver_);
    if (rc != 0) {
        std::string serr;
        serr.append("zmq_close is expected to return 0, ");
        serr.append("but the actual rc is ");
        serr.append(std::to_string(rc));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    for (auto &&it : senders_) {
        rc = zmq_setsockopt(it.second, ZMQ_LINGER, &linger, sizeof(linger));
        if (rc != 0 && errno != ETERM) {
            std::string serr;
            serr.append("zmq_setsockopt with ZMQ_LINGER is expected to return ");
            serr.append("0 or ETERM, but the actual rc is ");
            serr.append(std::to_string(rc));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        rc = zmq_close(it.second);
        if (rc != 0) {
            std::string serr;
            serr.append("zmq_close is expected to return 0, ");
            serr.append("but the actual rc is ");
            serr.append(std::to_string(rc));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }
    receiver_ = nullptr;
    senders_.clear();
    rc = zmq_ctx_destroy(context_);
    if (rc != 0) {
        std::string serr;
        serr.append("zmq_ctx_destroy is expected to return 0, ");
        serr.append("but the actual rc is ");
        serr.append(std::to_string(rc));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    context_ = nullptr;
}

int ZeroMQTransport::Bind(const NodeInfo &node, int maxRetry) {
    receiver_ = zmq_socket(context_, ZMQ_ROUTER);
    if (receiver_ == nullptr) {
        std::string serr;
        serr.append("Fail to create ZeroMQ receiver socket: ");
        serr.append(zmq_strerror(errno));
        serr.append("\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    int port = node.GetPort();
    unsigned seed = static_cast<unsigned>(time(nullptr) + port);
    for (int i = 0; i <= maxRetry; i++) {
        std::string addr = FormatActorAddress(node, port, true);
        if (zmq_bind(receiver_, addr.c_str()) == 0)
            break;
        if (i == maxRetry)
            port = -1;
        else
            port = 10000 + rand_r(&seed) % 40000;
    }
    if (port == -1) {
        std::string serr;
        serr.append("Fail to bind after retried ");
        serr.append(std::to_string(maxRetry));
        serr.append(" times.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    return port;
}

void ZeroMQTransport::Connect(const NodeInfo &node) {
    if (node.GetNodeId() == -1) {
        std::string serr = "Node id must not be -1.\n\n";
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (node.GetPort() == -1) {
        std::string serr = "Port must not be -1.\n\n";
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (node.GetHostName().empty()) {
        std::string serr = "Host name must not be empty.\n\n";
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    const int nodeId = node.GetNodeId();
    auto it = senders_.find(nodeId);
    if (it != senders_.end()) {
        int rc = zmq_close(it->second);
        if (rc != 0) {
            std::string serr;
            serr.append("zmq_close is expected to return 0, ");
            serr.append("but the actual rc is ");
            serr.append(std::to_string(rc));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }
    const NodeInfo &thisNode = GetConfig()->GetThisNodeInfo();
    // Worker doesn't connect to other workers and
    // server doesn't connect to other servers.
    // We may need to change this in the future.
    if (node.GetRole() == thisNode.GetRole() && node.GetNodeId() != thisNode.GetNodeId())
        return;
    void *sender = zmq_socket(context_, ZMQ_DEALER);
    if (sender == nullptr) {
        std::string serr;
        serr.append("Fail to create ZeroMQ sender socket: ");
        serr.append(zmq_strerror(errno));
        serr.append(" This often can be solved by \"sudo ulimit -n 65536\"");
        serr.append(" or edit /etc/security/limits.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (thisNode.GetNodeId() != -1) {
        std::string thisId = FormatActorIdentity(thisNode);
        zmq_setsockopt(sender, ZMQ_IDENTITY, thisId.data(), thisId.size());
    }
    std::string addr = FormatActorAddress(node, node.GetPort(), false);
    if (zmq_connect(sender, addr.c_str()) != 0) {
        std::string serr;
        serr.append("Fail to connect to ");
        serr.append(addr);
        serr.append(": ");
        serr.append(zmq_strerror(errno));
        serr.append("\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    senders_[nodeId] = sender;
}

int64_t ZeroMQTransport::SendMessage(const Message &msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int nodeId = msg.GetMessageMeta().GetReceiver();
    if (nodeId == -1) {
        std::string serr = "Receiver id must not be -1.\n\n";
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    auto it = senders_.find(nodeId);
    if (it == senders_.end()) {
        std::string serr;
        serr.append("There is no socket to node ");
        serr.append(NodeIdToString(nodeId));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    const NodeInfo &thisNode = GetConfig()->GetThisNodeInfo();
    void *const sender = it->second;
    auto metaPtr = new SmartArray<uint8_t>(msg.GetMessageMeta().PackAsThriftBuffer());
    const size_t metaSize = metaPtr->size();
    zmq_msg_t meta;
    zmq_msg_init_data(
        &meta, metaPtr->data(), metaSize,
        [](void *ptr, void *hint) {
            SmartArray<uint8_t> *metaPtr = static_cast<SmartArray<uint8_t> *>(hint);
            delete metaPtr;
        },
        metaPtr);
    const size_t n = msg.GetSlices().size();
    const int tag = n ? ZMQ_SNDMORE : 0;
    for (;;) {
        // Use `metaSize` instead of `metaPtr->size()`,
        // because when ZeroMQ sent the message successfully,
        // the lambda in `zmq_msg_init_data` may have been
        // called by ZeroMQ and then `metaPtr->size()` will
        // be some random number.
        if (zmq_msg_send(&meta, sender, tag) == metaSize)
            break;
        if (errno == EINTR) {
            spdlog::warn("{}: Interrupted while sending message meta to node {}.",
                         thisNode.ToShortString(), NodeIdToString(nodeId));
            continue;
        }
        // When sending is unsuccessfull, `zmq_msg_close` is not called by ZeroMQ;
        // When sending is successfull, `zmq_msg_close` will be called by ZeroMQ,
        // but call it again is not harmful.
        zmq_msg_close(&meta);
        std::string serr =
            fmt::format("{}: Fail to send message meta to node {}, errno: {} {}.\n\n{}",
                        thisNode.ToShortString(), NodeIdToString(nodeId), errno,
                        zmq_strerror(errno), GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    int64_t bytes = metaSize;
    zmq_msg_close(&meta);
    for (size_t i = 0; i < n; i++) {
        auto dataPtr = new SmartArray<uint8_t>(msg.GetSlice(i));
        const size_t dataSize = dataPtr->size();
        zmq_msg_t data;
        zmq_msg_init_data(
            &data, dataPtr->data(), dataSize,
            [](void *ptr, void *hint) {
                SmartArray<uint8_t> *dataPtr = static_cast<SmartArray<uint8_t> *>(hint);
                delete dataPtr;
            },
            dataPtr);
        const int tag = (i == n - 1) ? 0 : ZMQ_SNDMORE;
        for (;;) {
            // Use `dataSize` instead of `dataPtr->size()`,
            // because when ZeroMQ sent the message successfully,
            // the lambda in `zmq_msg_init_data` may have been
            // called by ZeroMQ and then `dataPtr->size()` will
            // be some random number.
            if (zmq_msg_send(&data, sender, tag) == dataSize)
                break;
            if (errno == EINTR) {
                spdlog::warn("{}: Interrupted while sending message slice to node [{}]. {}/{}",
                             thisNode.ToShortString(), nodeId, i, n);
                continue;
            }
            // When sending is unsuccessfull, `zmq_msg_close` is not called by ZeroMQ;
            // When sending is successfull, `zmq_msg_close` will be called by ZeroMQ,
            // but call it again is not harmful.
            zmq_msg_close(&data);
            std::string serr =
                fmt::format("{}: Fail to send message slice to node {}, errno: {} {}. {}/{}\n\n{}",
                            thisNode.ToShortString(), NodeIdToString(nodeId), errno,
                            zmq_strerror(errno), i, n, GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        bytes += dataSize;
        zmq_msg_close(&data);
    }
    return bytes;
}

int64_t ZeroMQTransport::ReceiveMessage(Message &msg) {
    const NodeInfo &thisNode = GetConfig()->GetThisNodeInfo();
    msg.ClearSlicesAndDataTypes();
    int64_t bytes = 0;
    for (size_t i = 0;; i++) {
        auto deleter = [](zmq_msg_t *msg) {
            zmq_msg_close(msg);
            delete msg;
        };
        std::unique_ptr<zmq_msg_t, decltype(deleter)> zmsg(new zmq_msg_t(), deleter);
        zmq_msg_init(zmsg.get());
        for (;;) {
            if (zmq_msg_recv(zmsg.get(), receiver_, 0) != -1)
                break;
            if (errno == EINTR) {
                spdlog::warn("{}: Interrupted while receiving message. i={}",
                             thisNode.ToShortString(), i);
                continue;
            }
            std::string serr = fmt::format("{}: Fail to receive message, errno: {} {}. i={}\n\n{}",
                                           thisNode.ToShortString(), errno, zmq_strerror(errno), i,
                                           GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        char *const buf = static_cast<char *>(zmq_msg_data(zmsg.get()));
        const size_t size = zmq_msg_size(zmsg.get());
        bytes += size;
        if (i == 0) {
            msg.GetMessageMeta().SetSender(ParseActorIdentity(buf, size));
            msg.GetMessageMeta().SetReceiver(GetConfig()->GetThisNodeInfo().GetNodeId());
            const int more = zmq_msg_more(zmsg.get());
            if (!more) {
                std::string serr;
                serr.append("zmq_msg_more is expected to return non-zero, but got ");
                serr.append(std::to_string(more));
                serr.append(".\n\n");
                serr.append(GetStackTrace());
                spdlog::error(serr);
                throw std::runtime_error(serr);
            }
        } else if (i == 1) {
            // The sender and receiver fields will be overridden by
            // UnpackFromThriftBuffer, save them and restore them later.
            const int sender = msg.GetMessageMeta().GetSender();
            const int receiver = msg.GetMessageMeta().GetReceiver();
            const uint8_t *const ptr = reinterpret_cast<const uint8_t *>(buf);
            msg.GetMessageMeta().UnpackFromThriftBuffer(ptr, size);
            msg.GetMessageMeta().SetSender(sender);
            msg.GetMessageMeta().SetReceiver(receiver);
            const int more = zmq_msg_more(zmsg.get());
            if (!more)
                break;
        } else {
            const int more = zmq_msg_more(zmsg.get());
            uint8_t *const ptr = reinterpret_cast<uint8_t *>(buf);
            auto slice = SmartArray<uint8_t>::Create(ptr, size, [zmsg = zmsg.get()](uint8_t *buf) {
                zmq_msg_close(zmsg);
                delete zmsg;
            });
            zmsg.release();
            msg.AddSlice(std::move(slice));
            if (!more)
                break;
        }
    }
    if (msg.GetSlices().size() != msg.GetMessageMeta().GetSliceDataTypes().size()) {
        std::string serr;
        serr.append("Corrupted message detected; meta indicates ");
        serr.append(std::to_string(msg.GetMessageMeta().GetSliceDataTypes().size()));
        serr.append(" slice(s), but found ");
        serr.append(std::to_string(msg.GetSlices().size()));
        serr.append(" in the body.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    return bytes;
}

std::string ZeroMQTransport::FormatActorAddress(const NodeInfo &node, int port,
                                                bool forServer) const {
    std::string hostName = node.GetHostName();
    if (forServer) {
        if (hostName.empty())
            hostName = "*";
        const bool useK8s = GetConfig()->UseKubernetes();
        if (useK8s && node.GetRole() == NodeRole::Coordinator)
            hostName = "0.0.0.0";
    }
    const bool isLocal = GetConfig()->IsLocalMode();
    std::string addr = isLocal ? "ipc:///tmp/" : "tcp://" + hostName + ":";
    std::string address = addr + std::to_string(port);
    return address;
}

std::string ZeroMQTransport::FormatActorIdentity(const NodeInfo &node) const {
    std::string id = "ps" + std::to_string(node.GetNodeId());
    return id;
}

int ZeroMQTransport::ParseActorIdentity(const char *buf, size_t size) const {
    if (size > 2 && buf[0] == 'p' && buf[1] == 's') {
        int id = 0;
        size_t i;
        for (i = 2; i < size; i++)
            if ('0' <= buf[i] && buf[i] <= '9')
                id = id * 10 + (buf[i] - '0');
            else
                break;
        if (i == size)
            return id;
    }
    return -1;
}

} // namespace metaspore
