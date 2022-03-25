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

#include <metaspore/node_control.h>

namespace metaspore {

std::string NodeControl::ToString() const { return ToJsonString(); }

std::string NodeControl::ToJsonString() const { return to_json().dump(); }

json11::Json NodeControl::to_json() const {
    std::vector<json11::Json> group;
    if (BarrierGroupContainsCoordinator())
        group.push_back("Coordinator");
    if (BarrierGroupContainsServers())
        group.push_back("Servers");
    if (BarrierGroupContainsWorkers())
        group.push_back("Workers");
    return json11::Json::object{
        {"command", NullableNodeControlCommandToString(command_)},
        {"nodes", nodes_},
        {"barrier_group", std::move(group)},
    };
}

} // namespace metaspore
