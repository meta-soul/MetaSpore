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

#include <metaspore/message_transport.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/zeromq_transport.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

MessageTransport::MessageTransport(std::shared_ptr<ActorConfig> config)
    : config_(std::move(config)) {}

std::unique_ptr<MessageTransport> MessageTransport::Create(std::shared_ptr<ActorConfig> config) {
    const std::string &type = config->GetTransportType();
    if (type == "ZeroMQ")
        return std::make_unique<ZeroMQTransport>(std::move(config));
    else {
        std::string serr;
        serr.append("MessageTransport type '");
        serr.append(type);
        serr.append("' is not supported.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
}

} // namespace metaspore
