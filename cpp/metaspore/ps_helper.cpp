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
#include <metaspore/network_utils.h>
#include <metaspore/ps_default_agent.h>
#include <metaspore/ps_helper.h>
#include <spdlog/spdlog.h>

namespace metaspore {

std::shared_ptr<metaspore::ActorConfig> GetLocalConfig(const std::string &role,
                                                       PSAgentCreator agent_creator) {
    auto config = std::make_shared<ActorConfig>();
    config->SetRootUri("localhost");
    config->SetRootPort(network_utils::get_available_port());
    config->SetServerCount(2);
    config->SetWorkerCount(2);
    // config->SetIsMessageDumpingEnabled(true);
    config->SetAgentCreator(std::move(agent_creator));
    if (role.empty()) {
        config->SetIsLocalMode(true);
    } else {
        if (role == "C") {
            config->SetNodeRole(NodeRole::Coordinator);
        } else if (role == "S") {
            config->SetNodeRole(NodeRole::Server);
        } else if (role == "W") {
            config->SetNodeRole(NodeRole::Worker);
        } else {
            std::cerr << "role must be [C | S | W]";
            exit(-1);
        }
    }
    return config;
}

std::shared_ptr<metaspore::ActorConfig> GetLocalConfig(const std::string &role) {
    return GetLocalConfig<PSDefaultAgent>(role);
}

} // namespace metaspore
