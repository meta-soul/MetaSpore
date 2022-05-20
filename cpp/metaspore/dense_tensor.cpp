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

#include <json11.hpp>
#include <metaspore/dense_tensor.h>
#include <metaspore/file_utils.h>
#include <metaspore/tensor_utils.h>
#include <string.h>

namespace metaspore {

void DenseTensor::Init(std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "DenseInit"},
        {"meta", GetMeta()},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void DenseTensor::Dispose(std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "DenseDispose"},
        {"name", GetMeta().GetName()},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void DenseTensor::Push(SmartArray<uint8_t> in, std::function<void()> cb, bool is_value,
                       bool is_state) {
    const size_t name_hash = GetMeta().GetNameHash();
    const size_t item_size = DataTypeToSize(GetMeta().GetDataType());
    const size_t slice_items =
        SliceElements(is_state ? GetMeta().GetStateShape() : GetMeta().GetDataShape());
    const size_t slice_length = item_size * slice_items;
    const int num_parts = GetMeta().GetPartitionCount();
    json11::Json json = json11::Json::object{
        {"command", "DensePush"},
        {"name", GetMeta().GetName()},
        {"is_value", is_value},
        {"is_state", is_state},
    };
    std::string command = json.dump();
    std::vector<PSMessage> reqs;
    reqs.reserve(num_parts);
    for (int k = 0; k < num_parts; k++) {
        PSMessage req = std::make_shared<Message>();
        req->GetMessageMeta().SetReceiver(ServerRankToNodeId(k));
        req->GetMessageMeta().SetBody(command);
        size_t begin = 0;
        size_t end = 0;
        GetMeta().ComputePartitionShapesWithHash(name_hash, k, begin, end, nullptr, nullptr);
        begin *= slice_length;
        end *= slice_length;
        SmartArray<uint8_t> k_in = in.Slice(begin, end);
        req->AddTypedSlice(k_in, GetMeta().GetDataType());
        reqs.push_back(req);
    }
    agent_->SendAllRequests(
        std::move(reqs), [cb](std::vector<PSMessage> reqs, std::vector<PSMessage> ress) { cb(); });
}

void DenseTensor::Pull(std::function<void(SmartArray<uint8_t> out)> cb, bool is_state) {
    json11::Json json = json11::Json::object{
        {"command", "DensePull"},
        {"name", GetMeta().GetName()},
        {"is_state", is_state},
    };
    PSMessage req = std::make_shared<Message>();
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [this, cb, is_state](PSMessage req, std::vector<PSMessage> ress) {
        const size_t name_hash = GetMeta().GetNameHash();
        const size_t item_size = DataTypeToSize(GetMeta().GetDataType());
        const size_t slice_items =
            SliceElements(is_state ? GetMeta().GetStateShape() : GetMeta().GetDataShape());
        const size_t slice_length = item_size * slice_items;
        const size_t total_items =
            TotalElements(is_state ? GetMeta().GetStateShape() : GetMeta().GetDataShape());
        const size_t total_length = item_size * total_items;
        const int num_parts = GetMeta().GetPartitionCount();
        SmartArray<uint8_t> out(total_length);
        for (int k = 0; k < num_parts; k++) {
            PSMessage res = ress.at(k);
            SmartArray<uint8_t> k_out = res->GetTypedSlice(0, GetMeta().GetDataType());
            const int sender = res->GetMessageMeta().GetSender();
            const int rank = NodeIdToRank(sender);
            size_t begin = 0;
            size_t end = 0;
            GetMeta().ComputePartitionShapesWithHash(name_hash, rank, begin, end, nullptr, nullptr);
            begin *= slice_length;
            end *= slice_length;
            memcpy(out.data() + begin, k_out.data(), end - begin);
        }
        cb(out);
    });
}

void DenseTensor::PushMeta(const DenseTensorMeta &meta, std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "DensePushMeta"},
        {"name", GetMeta().GetName()},
        {"meta", meta},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void DenseTensor::PullMeta(std::function<void(DenseTensorMeta meta)> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "DensePullMeta"},
        {"name", GetMeta().GetName()},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [this, cb](PSMessage req, std::vector<PSMessage> ress) {
        std::string body;
        int nodeId = -1;
        for (size_t k = 0; k < ress.size(); k++) {
            const std::string &body2 = ress.at(k)->GetMessageMeta().GetBody();
            if (body2.empty())
                continue;
            if (body.empty()) {
                body = body2;
                nodeId = ress.at(k)->GetMessageMeta().GetSender();
                continue;
            }
            if (body != body2) {
                const int nodeId2 = ress.at(k)->GetMessageMeta().GetSender();
                std::string serr;
                serr.append("Meta of dense tensor '");
                serr.append(GetMeta().GetName());
                serr.append("' on node ");
                serr.append(NodeIdToString(nodeId));
                serr.append(" and ");
                serr.append(NodeIdToString(nodeId2));
                serr.append(" mismatch.\n\n");
                serr.append(GetStackTrace());
                spdlog::error(serr);
                throw std::runtime_error(serr);
            }
        }
        if (body.empty())
            cb(GetMeta());
        else {
            DenseTensorMeta meta = DenseTensorMeta::FromJsonString(body);
            cb(std::move(meta));
        }
    });
}

void DenseTensor::Load(const std::string &dir_path, std::function<void()> cb, bool keep_meta) {
    std::string meta_path = GetDenseMetaPath(dir_path);
    std::string str = StreamReadAll(meta_path);
    DenseTensorMeta meta = DenseTensorMeta::FromJsonString(str);
    if (!meta.IsCompatible(meta_)) {
        std::string serr;
        serr.append("Incompatible meta detected in '");
        serr.append(meta_path);
        serr.append("', can not load dense tensor '");
        serr.append(GetMeta().GetName());
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    auto push_data_and_state = [this, dir_path, cb] {
        // To support sparse tensors repartition, ``if self.agent.rank == 0:`` is not checked in
        // Python code, and we must check this in C++ explicitly.
        if (agent_->GetAgentRank() != 0) {
            cb();
            return;
        }
        const size_t item_size = DataTypeToSize(GetMeta().GetDataType());
        const size_t total_items = TotalElements(GetMeta().GetDataShape());
        const size_t total_length = item_size * total_items;
        SmartArray<uint8_t> data(total_length);
        std::string data_path = GetDenseDataPath(dir_path);
        if (!data.empty() && LoadAsSArray(data_path, &data) < 0) {
            std::string serr;
            serr.append("Fail to load data file of dense tensor '");
            serr.append(GetMeta().GetName());
            serr.append("' from '");
            serr.append(data_path);
            serr.append("'.\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        Push(
            data,
            [this, dir_path, cb] {
                if (GetMeta().GetStateShape().empty()) {
                    cb();
                    return;
                }
                const size_t item_size = DataTypeToSize(GetMeta().GetDataType());
                const size_t total_items = TotalElements(GetMeta().GetStateShape());
                const size_t total_length = item_size * total_items;
                SmartArray<uint8_t> state(total_length);
                std::string state_path = GetDenseStatePath(dir_path);
                if (!state.empty() && LoadAsSArray(state_path, &state) < 0) {
                    std::string serr;
                    serr.append("Fail to load state file of dense tensor '");
                    serr.append(GetMeta().GetName());
                    serr.append("' from '");
                    serr.append(state_path);
                    serr.append("'.\n\n");
                    serr.append(GetStackTrace());
                    spdlog::error(serr);
                    throw std::runtime_error(serr);
                }
                Push(
                    state, [cb] { cb(); }, false, true);
            },
            true, false);
    };
    if (!keep_meta) {
        GetMeta().SetInitializerByData(meta.GetInitializerAsData());
        GetMeta().SetUpdaterByData(meta.GetUpdaterAsData());
    }
    if (keep_meta || agent_->GetAgentRank() != 0)
        push_data_and_state();
    else {
        meta.SetName(GetMeta().GetName());
        meta.SetPartitionCount(agent_->GetServerCount());
        PushMeta(meta, push_data_and_state);
    }
}

void DenseTensor::Save(const std::string &dir_path, std::function<void()> cb) {
    PullMeta([this, dir_path, cb](DenseTensorMeta meta) {
        std::string meta_path = GetDenseMetaPath(dir_path);
        std::string str = meta.ToJsonString();
        EnsureLocalDirectory(dir_path);
        StreamWriteAll(meta_path, str);
        Pull(
            [this, dir_path, cb](SmartArray<uint8_t> data) {
                std::string data_path = GetDenseDataPath(dir_path);
                if (!data.empty() && SaveAsSArray(data_path, data) < 0) {
                    std::string serr;
                    serr.append("Fail to save data file of dense tensor ");
                    serr.append(GetMeta().GetName());
                    serr.append("' to '");
                    serr.append(data_path);
                    serr.append("'.\n\n");
                    serr.append(GetStackTrace());
                    spdlog::error(serr);
                    throw std::runtime_error(serr);
                }
                Pull(
                    [this, dir_path, cb](SmartArray<uint8_t> state) {
                        std::string state_path = GetDenseStatePath(dir_path);
                        if (!state.empty() && SaveAsSArray(state_path, state) < 0) {
                            std::string serr;
                            serr.append("Fail to save state file of dense tensor '");
                            serr.append(GetMeta().GetName());
                            serr.append("' to '");
                            serr.append(state_path);
                            serr.append("'.\n\n");
                            serr.append(GetStackTrace());
                            spdlog::error(serr);
                            throw std::runtime_error(serr);
                        }
                        cb();
                    },
                    true);
            },
            false);
    });
}

std::string DenseTensor::GetDenseMetaPath(const std::string &dir_path) const {
    std::string file_name = fmt::format("{}__dense_meta.json", GetMeta().GetName());
    std::string file_path = JoinPath(dir_path, file_name);
    return file_path;
}

std::string DenseTensor::GetDenseDataPath(const std::string &dir_path) const {
    std::string file_name = fmt::format("{}__dense_data.dat", GetMeta().GetName());
    std::string file_path = JoinPath(dir_path, file_name);
    return file_path;
}

std::string DenseTensor::GetDenseStatePath(const std::string &dir_path) const {
    std::string file_name = fmt::format("{}__dense_state.dat", GetMeta().GetName());
    std::string file_path = JoinPath(dir_path, file_name);
    return file_path;
}

} // namespace metaspore
