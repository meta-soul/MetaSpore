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

#include <metaspore/actor_config.h>
#include <metaspore/ps_agent.h>

namespace metaspore {

std::shared_ptr<metaspore::ActorConfig> GetLocalConfig(const std::string &role,
                                                       PSAgentCreator agent_creator);

template <typename T>
std::shared_ptr<metaspore::ActorConfig> GetLocalConfig(const std::string &role) {
    return GetLocalConfig(role, [] {
        std::shared_ptr<PSAgent> agent = std::make_shared<T>();
        return agent;
    });
}

std::shared_ptr<metaspore::ActorConfig> GetLocalConfig(const std::string &role);

} // namespace metaspore
