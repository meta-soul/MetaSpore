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

#include <metaspore/node_encoding.h>
#include <metaspore/node_info.h>
#include <sstream>

namespace metaspore {

std::string NodeInfo::ToString() const { return ToJsonString(); }

std::string NodeInfo::ToShortString() const {
    std::ostringstream sout;
    switch (role_) {
    case NodeRole::Coordinator:
        sout << "C";
        break;
    case NodeRole::Server:
        sout << "S";
        break;
    case NodeRole::Worker:
        sout << "W";
        break;
    default:
        sout << "?";
        break;
    }
    if (node_id_ != -1)
        sout << "[" << NodeIdToRank(node_id_) << "]";
    sout << ":" << node_id_;
    return sout.str();
}

std::string NodeInfo::ToJsonString() const { return to_json().dump(); }

json11::Json NodeInfo::to_json() const {
    return json11::Json::object{
        {"role", NullableNodeRoleToString(role_)},
        {"node_id", node_id_},
        {"host_name", host_name_},
        {"port", port_},
    };
}

} // namespace metaspore
