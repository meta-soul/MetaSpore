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

#include <metaspore/node_role.h>

//
// ``node_encoding.h`` defines integer encodings of nodes and
// node groups and related functions.
//

namespace metaspore {

// Integer encodings of node group of the same role.
#undef METASPORE_NODE_ROLE_DEF
#define METASPORE_NODE_ROLE_DEF(n) constexpr int n##Group = 1 << static_cast<int>(NodeRole::n);
METASPORE_NODE_ROLES(METASPORE_NODE_ROLE_DEF)

// Tag value specify that this integer encoding identify a single node.
constexpr int SingleNodeIdTag = 1 << (static_cast<int>(NodeRole::Worker) + 1);

// Tag value specify that this integer encoding identify a node of the specific role.
#undef METASPORE_NODE_ROLE_DEF
#define METASPORE_NODE_ROLE_DEF(n)                                                                 \
    constexpr int n##NodeIdTag = 1 << static_cast<int>(NodeRole::n) | SingleNodeIdTag;
METASPORE_NODE_ROLES(METASPORE_NODE_ROLE_DEF)

// Encode node numbered ``rank`` as a node id, which is an integer. ``rank`` is zero-based.
#undef METASPORE_NODE_ROLE_DEF
#define METASPORE_NODE_ROLE_DEF(n)                                                                 \
    constexpr int n##RankToNodeId(int rank) { return rank << 4 | n##NodeIdTag; }
METASPORE_NODE_ROLES(METASPORE_NODE_ROLE_DEF)

// Get the zero-based ``rank`` from node id ``id``.
constexpr int NodeIdToRank(int id) { return id >> 4; }

// Node id of the coordinator node. Since there is a single coordinator node
// in the Parameter Server system, its node id can be pre-computed.
constexpr int CoordinatorNodeId = CoordinatorRankToNodeId(0);

// Convert integer node id to a descriptive string.
std::string NodeIdToString(int node_id);

} // namespace metaspore
