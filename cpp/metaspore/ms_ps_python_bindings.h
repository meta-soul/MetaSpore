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

#include <metaspore/ps_agent.h>

namespace metaspore {

template <class PSAgentBase = metaspore::PSAgent> class PyPSAgent : public PSAgentBase {
  public:
    using PSAgentBase::PSAgentBase;

    void Run() override { PYBIND11_OVERLOAD_NAME(void, PSAgentBase, "run", Run, ); }

    void HandleRequest(metaspore::PSMessage req) override {
        PYBIND11_OVERLOAD_NAME(void, PSAgentBase, "handle_request", HandleRequest, req);
    }

    void Finalize() override { PYBIND11_OVERLOAD_NAME(void, PSAgentBase, "finalize", Finalize, ); }
};

} // namespace metaspore
