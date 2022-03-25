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

#include <string>

//
// ``node_contrl_command.h`` defines enum ``NodeControlCommand`` to represent
// control commands sent between Parameter Server nodes.
//

namespace metaspore {

//
// Use the X Macro technique to simplify code. See the following page
// for more information about X Macros:
//
//   https://en.wikipedia.org/wiki/X_Macro
//

#define METASPORE_NODE_CONTROL_COMMANDS(X)                                                         \
    X(Terminate)                                                                                   \
    X(AddNode)                                                                                     \
    X(Barrier)                                                                                     \
    /**/

enum class NodeControlCommand {
#undef METASPORE_NODE_CONTROL_COMMAND_DEF
#define METASPORE_NODE_CONTROL_COMMAND_DEF(n) n,
    METASPORE_NODE_CONTROL_COMMANDS(METASPORE_NODE_CONTROL_COMMAND_DEF)
};

// A missing ``NodeControlCommand`` is represented by ``NodeControlCommand(-1)``.
constexpr NodeControlCommand NullNodeControlCommand = static_cast<NodeControlCommand>(-1);
constexpr const char *NullNodeControlCommandString = "null";

// Functions to convert ``NodeControlCommand`` to and from strings.
std::string NodeControlCommandToString(NodeControlCommand command);
NodeControlCommand NodeControlCommandFromString(const std::string &str);

std::string NullableNodeControlCommandToString(NodeControlCommand command);
NodeControlCommand NullableNodeControlCommandFromString(const std::string &str);

} // namespace metaspore
