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

#include <future>
#include <iostream>
#include <json11.hpp>
#include <metaspore/debug.h>
#include <metaspore/ps_default_agent.h>
#include <metaspore/stack_trace_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

#define PS_DEFAULT_AGENT_COMMANDS(X)                                                               \
    X(DenseInit)                                                                                   \
    X(DenseDispose)                                                                                \
    X(DensePush)                                                                                   \
    X(DensePull)                                                                                   \
    X(DensePushMeta)                                                                               \
    X(DensePullMeta)                                                                               \
    X(SparseInit)                                                                                  \
    X(SparseDispose)                                                                               \
    X(SparseClear)                                                                                 \
    X(SparsePush)                                                                                  \
    X(SparsePull)                                                                                  \
    X(SparsePushPartition)                                                                         \
    X(SparsePullPartition)                                                                         \
    X(SparsePushMeta)                                                                              \
    X(SparsePullMeta)                                                                              \
    X(SparseLoad)                                                                                  \
    X(SparseSave)                                                                                  \
    X(SparseExport)                                                                                \
    X(SparsePruneSmall)                                                                            \
    X(SparsePruneOld)                                                                              \
    /**/

enum class PSDefaultAgentCommand {
#undef PS_DEFAULT_AGENT_COMMAND_DEF
#define PS_DEFAULT_AGENT_COMMAND_DEF(n) n,
    PS_DEFAULT_AGENT_COMMANDS(PS_DEFAULT_AGENT_COMMAND_DEF)
};

static const std::unordered_map<std::string, PSDefaultAgentCommand> PSDefaultAgentCommandMap = {
#undef PS_DEFAULT_AGENT_COMMAND_DEF
#define PS_DEFAULT_AGENT_COMMAND_DEF(n) {#n, PSDefaultAgentCommand::n},
    PS_DEFAULT_AGENT_COMMANDS(PS_DEFAULT_AGENT_COMMAND_DEF)};

void PSDefaultAgent::Run() {
    pybind11::gil_scoped_acquire gil;
    auto method = py_agent_.attr("run");
    method();
}

void PSDefaultAgent::HandleRequest(PSMessage req) {
    if (!IsServer()) {
        pybind11::gil_scoped_acquire gil;
        auto method = py_agent_.attr("handle_request");
        method(req);
        return;
    }
    if (!store_) {
        store_ = std::make_unique<TensorPartitionStore>();
        store_->SetPartitionCount(GetServerCount());
        store_->SetPartitionIndex(GetAgentRank());
    }
    std::string err;
    const std::string &str = req->GetMessageMeta().GetBody();
    // std::cout << "str: " << str << std::endl;
    json11::Json json = json11::Json::parse(str, err);
    if (!err.empty()) {
        std::string serr;
        serr.append("Unable to parse PSDefaultAgent command from JSON string; str: ");
        serr.append(str);
        serr.append(", err: ");
        serr.append(err);
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    const std::string &command = json["command"].string_value();
    // std::cout << "command: " << command << ", str: " << str << std::endl;
    auto it = PSDefaultAgentCommandMap.find(command);
    if (it == PSDefaultAgentCommandMap.end()) {
        std::string serr;
        serr.append("Unknown PSDefaultAgent command '");
        serr.append(command);
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    switch (it->second) {
    case PSDefaultAgentCommand::DenseInit: {
        DenseTensorMeta meta = DenseTensorMeta::FromJson(json["meta"]);
        store_->DenseInit(meta);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::DenseDispose: {
        const std::string &name = json["name"].string_value();
        store_->DenseDispose(name);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::DensePush: {
        const std::string &name = json["name"].string_value();
        const bool is_value = json["is_value"].bool_value();
        const bool is_state = json["is_state"].bool_value();
        store_->DensePush(name, req, is_value, is_state);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::DensePull: {
        const std::string &name = json["name"].string_value();
        const bool is_state = json["is_state"].bool_value();
        PSMessage res = store_->DensePull(name, is_state);
        SendResponse(req, res);
        break;
    }
    case PSDefaultAgentCommand::DensePushMeta: {
        const std::string &name = json["name"].string_value();
        DenseTensorMeta meta = DenseTensorMeta::FromJson(json["meta"]);
        store_->DensePushMeta(name, meta);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::DensePullMeta: {
        const std::string &name = json["name"].string_value();
        PSMessage res = store_->DensePullMeta(name);
        SendResponse(req, res);
        break;
    }
    case PSDefaultAgentCommand::SparseInit: {
        SparseTensorMeta meta = SparseTensorMeta::FromJson(json["meta"]);
        store_->SparseInit(meta);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparseDispose: {
        const std::string &name = json["name"].string_value();
        store_->SparseDispose(name);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparseClear: {
        const std::string &name = json["name"].string_value();
        store_->SparseClear(name);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparsePush: {
        const std::string &name = json["name"].string_value();
        const bool is_value = json["is_value"].bool_value();
        store_->SparsePush(name, req, is_value);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparsePull: {
        const std::string &name = json["name"].string_value();
        const bool read_only = json["read_only"].bool_value();
        const bool nan_fill = json["nan_fill"].bool_value();
        PSMessage res = store_->SparsePull(name, req, read_only, nan_fill);
        SendResponse(req, res);
        break;
    }
    case PSDefaultAgentCommand::SparsePushPartition: {
        const std::string &name = json["name"].string_value();
        const bool data_only = json["data_only"].bool_value();
        const bool skip_existing = json["skip_existing"].bool_value();
        store_->SparsePushPartition(name, req, data_only, skip_existing);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparsePullPartition: {
        const std::string &name = json["name"].string_value();
        const bool data_only = json["data_only"].bool_value();
        const int index = json["index"].int_value();
        const int count = json["count"].int_value();
        PSMessage res = store_->SparsePullPartition(name, data_only, index, count);
        SendResponse(req, res);
        break;
    }
    case PSDefaultAgentCommand::SparsePushMeta: {
        const std::string &name = json["name"].string_value();
        SparseTensorMeta meta = SparseTensorMeta::FromJson(json["meta"]);
        store_->SparsePushMeta(name, meta);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparsePullMeta: {
        const std::string &name = json["name"].string_value();
        PSMessage res = store_->SparsePullMeta(name);
        SendResponse(req, res);
        break;
    }
    case PSDefaultAgentCommand::SparseLoad: {
        const std::string &name = json["name"].string_value();
        const std::string &dir_path = json["dir_path"].string_value();
        store_->SparseLoad(name, dir_path);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparseSave: {
        const std::string &name = json["name"].string_value();
        const std::string &dir_path = json["dir_path"].string_value();
        const bool text_mode = json["text_mode"].bool_value();
        store_->SparseSave(name, dir_path, text_mode);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparseExport: {
        const std::string &name = json["name"].string_value();
        const std::string &dir_path = json["dir_path"].string_value();
        store_->SparseExport(name, dir_path);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparsePruneSmall: {
        const std::string &name = json["name"].string_value();
        const double epsilon = json["epsilon"].number_value();
        store_->SparsePruneSmall(name, epsilon);
        PSAgent::HandleRequest(req);
        break;
    }
    case PSDefaultAgentCommand::SparsePruneOld: {
        const std::string &name = json["name"].string_value();
        const int max_age = json["max_age"].int_value();
        store_->SparsePruneOld(name, max_age);
        PSAgent::HandleRequest(req);
        break;
    }
    default: {
        std::string serr;
        serr.append("Unimplemented PSDefaultAgent command '");
        serr.append(command);
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    }
}

void PSDefaultAgent::Finalize() {
    // Call the ``_finalize`` method of the Python agent object to remove its
    // reference to this C++ agent object and then remove the reference to the
    // Python agent object. This breaks the reference cycle.
    pybind11::gil_scoped_acquire gil;
    auto method = py_agent_.attr("_finalize");
    method();
    py_agent_ = pybind11::object();
}

} // namespace metaspore
