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

#include <json11.hpp>
#include <metaspore/node_role.h>
#include <string>
#include <utility>

//
// ``node_info.h`` defines class ``NodeInfo`` which stores
// information about nodes in the Parameter Server system.
//

namespace metaspore {

class NodeInfo {
  public:
    NodeRole GetRole() const { return role_; }
    void SetRole(NodeRole value) { role_ = value; }

    int GetNodeId() const { return node_id_; }
    void SetNodeId(int value) { node_id_ = value; }

    const std::string &GetHostName() const { return host_name_; }
    void SetHostName(std::string value) { host_name_ = std::move(value); }

    int GetPort() const { return port_; }
    void SetPort(int value) { port_ = value; }

    std::string GetAddress() const { return host_name_ + ":" + std::to_string(port_); }

    std::string ToString() const;
    std::string ToShortString() const;
    std::string ToJsonString() const;
    json11::Json to_json() const;

  private:
    NodeRole role_ = NullNodeRole;
    int node_id_ = -1;
    std::string host_name_;
    int port_ = -1;
};

} // namespace metaspore
