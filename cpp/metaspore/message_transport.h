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
#include <metaspore/actor_config.h>
#include <metaspore/message.h>
#include <metaspore/node_info.h>
#include <utility>

namespace metaspore {

class MessageTransport {
  public:
    explicit MessageTransport(std::shared_ptr<ActorConfig> config);
    virtual ~MessageTransport() {}

    std::shared_ptr<ActorConfig> GetConfig() const { return config_; }
    void SetConfig(std::shared_ptr<ActorConfig> value) { config_ = std::move(value); }

    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual int Bind(const NodeInfo &node, int maxRetry) = 0;
    virtual void Connect(const NodeInfo &node) = 0;
    virtual int64_t SendMessage(const Message &msg) = 0;
    virtual int64_t ReceiveMessage(Message &msg) = 0;

    static std::unique_ptr<MessageTransport> Create(std::shared_ptr<ActorConfig> config);

  private:
    std::shared_ptr<ActorConfig> config_;
};

} // namespace metaspore
