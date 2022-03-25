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

#include <metaspore/message_transport.h>
#include <mutex>
#include <unordered_map>

namespace metaspore {

class ZeroMQTransport : public MessageTransport {
  public:
    explicit ZeroMQTransport(std::shared_ptr<ActorConfig> config);

    void Start() override;
    void Stop() override;
    int Bind(const NodeInfo &node, int maxRetry) override;
    void Connect(const NodeInfo &node) override;
    int64_t SendMessage(const Message &msg) override;
    int64_t ReceiveMessage(Message &msg) override;

  private:
    std::string FormatActorAddress(const NodeInfo &node, int port, bool forServer) const;
    std::string FormatActorIdentity(const NodeInfo &node) const;
    int ParseActorIdentity(const char *buf, size_t size) const;

    std::mutex mutex_;
    void *context_ = nullptr;
    void *receiver_ = nullptr;
    std::unordered_map<int, void *> senders_;
};

} // namespace metaspore
