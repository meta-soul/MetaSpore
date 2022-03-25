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
#include <sstream>

namespace metaspore {

std::string NodeIdToString(int node_id) {
    std::ostringstream sout;
    if (node_id & CoordinatorGroup)
        sout << "C";
    else if (node_id & ServerGroup)
        sout << "S";
    else if (node_id & WorkerGroup)
        sout << "W";
    else
        sout << "?";
    if (node_id & SingleNodeIdTag)
        sout << "[" << NodeIdToRank(node_id) << "]";
    else
        sout << "*";
    sout << ":" << node_id;
    return sout.str();
}

} // namespace metaspore
