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

#include <metaspore/node_control_command.h>
#include <metaspore/node_encoding.h>
#include <metaspore/node_info.h>
#include <stdint.h>
#include <utility>
#include <vector>

//
// ``node_control.h`` defines class ``NodeControl`` which contains information
// about node control messages.
//

namespace metaspore {

class NodeControl {
  public:
    bool IsEmpty() const { return command_ == NullNodeControlCommand; }

    NodeControlCommand GetCommand() const { return command_; }
    void SetCommand(NodeControlCommand value) { command_ = value; }

    //
    // Methods related to node info.
    //
    std::vector<NodeInfo> &GetNodes() { return nodes_; }
    const std::vector<NodeInfo> &GetNodes() const { return nodes_; }
    void SetNodes(std::vector<NodeInfo> value) { nodes_ = std::move(value); }

    void ClearNodes() { nodes_.clear(); }
    void AddNode(NodeInfo value) { nodes_.push_back(std::move(value)); }

    //
    // Methods related to barrier group.
    //
    int GetBarrierGroup() const { return barrier_group_; }
    void SetBarrierGroup(int value) { barrier_group_ = value; }

    void ClearBarrierGroup() { barrier_group_ = 0; }
    void AddCoordinatorToBarrierGroup() { barrier_group_ |= CoordinatorGroup; }
    void AddServersToBarrierGroup() { barrier_group_ |= ServerGroup; }
    void AddWorkersToBarrierGroup() { barrier_group_ |= WorkerGroup; }
    void RemoveCoordinatorFromBarrierGroup() { barrier_group_ &= ~CoordinatorGroup; }
    void RemoveServersFromBarrierGroup() { barrier_group_ &= ~ServerGroup; }
    void RemoveWorkersFromBarrierGroup() { barrier_group_ &= ~WorkerGroup; }
    bool BarrierGroupContainsCoordinator() const {
        return (barrier_group_ & CoordinatorGroup) != 0;
    }
    bool BarrierGroupContainsServers() const { return (barrier_group_ & ServerGroup) != 0; }
    bool BarrierGroupContainsWorkers() const { return (barrier_group_ & WorkerGroup) != 0; }

    std::string ToString() const;
    std::string ToJsonString() const;
    json11::Json to_json() const;

  private:
    NodeControlCommand command_ = NullNodeControlCommand;
    std::vector<NodeInfo> nodes_;
    int barrier_group_ = 0;
};

} // namespace metaspore
