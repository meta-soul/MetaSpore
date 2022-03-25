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
#include <metaspore/tensor_partition_store.h>
#include <pybind11/pybind11.h>

namespace metaspore {

class __attribute__((visibility("hidden"))) PSDefaultAgent : public PSAgent {
  public:
    pybind11::object GetPyAgent() const { return py_agent_; }
    void SetPyAgent(pybind11::object value) { py_agent_ = std::move(value); }

    void Run() override;
    void HandleRequest(PSMessage req) override;
    void Finalize() override;

  private:
    pybind11::object py_agent_;
    std::unique_ptr<TensorPartitionStore> store_;
};

} // namespace metaspore
