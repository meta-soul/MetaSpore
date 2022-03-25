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
#include <metaspore/array_hash_map_reader.h>
#include <metaspore/debug.h>
#include <metaspore/io.h>
#include <metaspore/sparse_tensor.h>
#include <string.h>

namespace metaspore {

void SparseTensor::Init(std::function<void()> cb) {
    GetMeta().CheckSparseTensorMeta(0);
    GetMeta().ComputeSliceInfo();
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparseInit"},
        {"meta", GetMeta()},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void SparseTensor::Dispose(std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparseDispose"},
        {"name", GetMeta().GetName()},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void SparseTensor::Clear(std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparseClear"},
        {"name", GetMeta().GetName()},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void SparseTensor::Push(SmartArray<uint8_t> keys, SmartArray<uint8_t> in, std::function<void()> cb,
                        bool is_value) {
    const size_t index_count = keys.size() / sizeof(uint64_t);
    const uint64_t *const indices = reinterpret_cast<uint64_t *>(keys.data());
    const uint8_t *source = in.data();
    const size_t num_parts = GetMeta().GetPartitionCount();
    std::vector<std::vector<uint64_t>> part_keys(num_parts);
    std::vector<std::vector<uint8_t>> part_data(num_parts);
    for (size_t i = 0; i < index_count; i++) {
        const uint64_t key = indices[i];
        const size_t part = key % num_parts;
        part_keys.at(part).push_back(key);
        VectorAppend(part_data.at(part), source, GetMeta().GetSliceDataLength());
        source += GetMeta().GetSliceDataLength();
    }
    json11::Json json = json11::Json::object{
        {"command", "SparsePush"},
        {"name", GetMeta().GetName()},
        {"is_value", is_value},
    };
    std::string command = json.dump();
    std::vector<PSMessage> reqs;
    reqs.reserve(num_parts);
    for (size_t k = 0; k < num_parts; k++) {
        PSMessage req = std::make_shared<Message>();
        req->GetMessageMeta().SetReceiver(ServerRankToNodeId(k));
        req->GetMessageMeta().SetBody(command);
        auto k_keys = SmartArray<uint64_t>::Wrap(std::move(part_keys.at(k)));
        auto k_in = SmartArray<uint8_t>::Wrap(std::move(part_data.at(k)));
        req->AddTypedSlice(k_keys);
        req->AddTypedSlice(k_in, GetMeta().GetDataType());
        reqs.push_back(req);
    }
    agent_->SendAllRequests(
        std::move(reqs), [cb](std::vector<PSMessage> reqs, std::vector<PSMessage> ress) { cb(); });
}

void SparseTensor::Pull(SmartArray<uint8_t> keys, std::function<void(SmartArray<uint8_t> out)> cb,
                        bool read_only, bool nan_fill) {
    const size_t index_count = keys.size() / sizeof(uint64_t);
    const uint64_t *const indices = reinterpret_cast<uint64_t *>(keys.data());
    const size_t num_parts = GetMeta().GetPartitionCount();
    std::vector<std::vector<uint64_t>> part_keys(num_parts);
    for (size_t i = 0; i < index_count; i++) {
        const uint64_t key = indices[i];
        const size_t part = key % num_parts;
        part_keys.at(part).push_back(key);
    }
    json11::Json json = json11::Json::object{
        {"command", "SparsePull"},
        {"name", GetMeta().GetName()},
        {"read_only", read_only},
        {"nan_fill", nan_fill},
    };
    std::string command = json.dump();
    std::vector<PSMessage> reqs;
    reqs.reserve(num_parts);
    for (size_t k = 0; k < num_parts; k++) {
        PSMessage req = std::make_shared<Message>();
        req->GetMessageMeta().SetReceiver(ServerRankToNodeId(k));
        req->GetMessageMeta().SetBody(command);
        auto k_keys = SmartArray<uint64_t>::Wrap(std::move(part_keys.at(k)));
        req->AddTypedSlice(k_keys);
        reqs.push_back(req);
    }
    agent_->SendAllRequests(std::move(reqs), [this, keys, cb](std::vector<PSMessage> reqs,
                                                              std::vector<PSMessage> ress) {
        const size_t index_count = keys.size() / sizeof(uint64_t);
        SmartArray<uint8_t> out(index_count * GetMeta().GetSliceDataLength());
        uint8_t *target = out.data();
        const uint64_t *const indices = reinterpret_cast<const uint64_t *>(keys.data());
        const size_t num_parts = GetMeta().GetPartitionCount();
        std::vector<const uint8_t *> sources(num_parts);
        for (size_t k = 0; k < ress.size(); k++) {
            PSMessage res = ress.at(k);
            const int sender = res->GetMessageMeta().GetSender();
            const int rank = NodeIdToRank(sender);
            SmartArray<uint8_t> k_out = res->GetTypedSlice(0, GetMeta().GetDataType());
            const uint8_t *const source = k_out.data();
            sources.at(rank) = source;
        }
        for (size_t i = 0; i < index_count; i++) {
            const uint64_t key = indices[i];
            const size_t part = key % num_parts;
            const uint8_t *&source = sources.at(part);
            memcpy(target, source, GetMeta().GetSliceDataLength());
            target += GetMeta().GetSliceDataLength();
            source += GetMeta().GetSliceDataLength();
        }
        cb(out);
    });
}

void SparseTensor::PushPartition(ArrayHashMap<uint64_t, uint8_t> &data, std::function<void()> cb,
                                 bool data_only, bool skip_existing) {
    const size_t index_count = data.size();
    const uint64_t *const indices = data.get_keys_array();
    const uint8_t *source = data.get_values_array();
    const size_t num_parts = GetMeta().GetPartitionCount();
    const size_t vec_length =
        data_only ? GetMeta().GetSliceDataLength() : GetMeta().GetSliceTotalBytes();
    std::vector<std::vector<uint64_t>> part_keys(num_parts);
    std::vector<std::vector<uint8_t>> part_data(num_parts);
    for (size_t i = 0; i < index_count; i++) {
        const uint64_t key = indices[i];
        const size_t part = key % num_parts;
        part_keys.at(part).push_back(key);
        VectorAppend(part_data.at(part), source, vec_length);
        source += vec_length;
    }
    json11::Json json = json11::Json::object{
        {"command", "SparsePushPartition"},
        {"name", GetMeta().GetName()},
        {"data_only", data_only},
        {"skip_existing", skip_existing},
    };
    std::string command = json.dump();
    std::vector<PSMessage> reqs;
    reqs.reserve(num_parts);
    for (size_t k = 0; k < num_parts; k++) {
        PSMessage req = std::make_shared<Message>();
        req->GetMessageMeta().SetReceiver(ServerRankToNodeId(k));
        req->GetMessageMeta().SetBody(command);
        auto k_keys = SmartArray<uint64_t>::Wrap(std::move(part_keys.at(k)));
        auto k_in = SmartArray<uint8_t>::Wrap(std::move(part_data.at(k)));
        req->AddTypedSlice(k_keys);
        req->AddTypedSlice(k_in, GetMeta().GetDataType());
        reqs.push_back(req);
    }
    agent_->SendAllRequests(
        std::move(reqs), [cb](std::vector<PSMessage> reqs, std::vector<PSMessage> ress) { cb(); });
}

void SparseTensor::PullPartition(ArrayHashMap<uint64_t, uint8_t> &data, std::function<void()> cb,
                                 bool data_only, int index, int count) {
    if (index == -1)
        index = agent_->GetAgentRank();
    if (count == -1)
        count = agent_->GetWorkerCount();
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparsePullPartition"},
        {"name", GetMeta().GetName()},
        {"data_only", data_only},
        {"index", index},
        {"count", count},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    const size_t vec_length =
        data_only ? GetMeta().GetSliceDataLength() : GetMeta().GetSliceTotalBytes();
    agent_->BroadcastRequest(
        req, [this, &data, cb, vec_length](PSMessage req, std::vector<PSMessage> ress) {
            for (size_t k = 0; k < ress.size(); k++) {
                PSMessage res = ress.at(k);
                SmartArray<uint8_t> k_keys = res->GetTypedSlice<uint64_t>(0).Cast<uint8_t>();
                SmartArray<uint8_t> k_values = res->GetTypedSlice(1, GetMeta().GetDataType());
                const size_t index_count = k_keys.size() / sizeof(uint64_t);
                const uint64_t *const indices = reinterpret_cast<const uint64_t *>(k_keys.data());
                const uint8_t *source = k_values.data();
                for (size_t i = 0; i < index_count; i++) {
                    const uint64_t key = indices[i];
                    uint8_t *const target = data.get_or_init(key);
                    memcpy(target, source, vec_length);
                    source += vec_length;
                }
            }
            cb();
        });
}

void SparseTensor::PushMeta(const SparseTensorMeta &meta, std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparsePushMeta"},
        {"name", GetMeta().GetName()},
        {"meta", meta},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void SparseTensor::PullMeta(std::function<void(SparseTensorMeta meta)> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparsePullMeta"},
        {"name", GetMeta().GetName()},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [this, cb](PSMessage req, std::vector<PSMessage> ress) {
        for (size_t k = 0; k < ress.size(); k++) {
            const std::string &body1 = ress.at(0)->GetMessageMeta().GetBody();
            const std::string &body2 = ress.at(k)->GetMessageMeta().GetBody();
            if (body1 != body2) {
                const int nodeId1 = ress.at(0)->GetMessageMeta().GetSender();
                const int nodeId2 = ress.at(k)->GetMessageMeta().GetSender();
                std::string serr;
                serr.append("Meta of sparse tensor '");
                serr.append(GetMeta().GetName());
                serr.append("' on node ");
                serr.append(NodeIdToString(nodeId1));
                serr.append(" and ");
                serr.append(NodeIdToString(nodeId2));
                serr.append(" mismatch.\n\n");
                serr.append(GetStackTrace());
                spdlog::error(serr);
                throw std::runtime_error(serr);
            }
        }
        const std::string &body = ress.at(0)->GetMessageMeta().GetBody();
        SparseTensorMeta meta = SparseTensorMeta::FromJsonString(body);
        cb(std::move(meta));
    });
}

void SparseTensor::Load(const std::string &dir_path, std::function<void()> cb, bool keep_meta) {
    std::string meta_path = GetSparseMetaPath(dir_path);
    std::string str = StreamReadAll(meta_path);
    SparseTensorMeta meta = SparseTensorMeta::FromJsonString(str);
    if (!meta.IsCompatible(meta_)) {
        std::string serr;
        serr.append("Incompatible meta detected in '");
        serr.append(meta_path);
        serr.append("', can not load sparse tensor '");
        serr.append(GetMeta().GetName());
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    const int old_part_count = meta.GetPartitionCount();
    auto load_data_and_state = [this, dir_path, cb, old_part_count] {
        if (GetMeta().GetPartitionCount() == old_part_count) {
            // To support sparse tensors repartition, ``if self.agent.rank == 0:`` is not checked in
            // Python code, and we must check this in C++ explicitly.
            if (agent_->GetAgentRank() != 0) {
                cb();
                return;
            }
            // Letting worker #0 to send the request is enough.
            PSMessage req = std::make_shared<Message>();
            json11::Json json = json11::Json::object{
                {"command", "SparseLoad"},
                {"name", GetMeta().GetName()},
                {"dir_path", dir_path},
            };
            req->GetMessageMeta().SetReceiver(ServerGroup);
            req->GetMessageMeta().SetBody(json.dump());
            agent_->BroadcastRequest(req,
                                     [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
        } else {
            // The sparse tensor is repartitioned, all workers need to
            // execute the following logic.
            std::string meta_file_path = GetSparseMetaPath(dir_path);
            ImportFrom(meta_file_path, cb, false, false, false, "");
        }
    };
    if (!keep_meta) {
        GetMeta().SetInitializerByData(meta.GetInitializerAsData());
        GetMeta().SetUpdaterByData(meta.GetUpdaterAsData());
    }
    if (keep_meta || agent_->GetAgentRank() != 0)
        load_data_and_state();
    else {
        meta.SetName(GetMeta().GetName());
        meta.SetPartitionCount(agent_->GetServerCount());
        PushMeta(meta, load_data_and_state);
    }
}

void SparseTensor::Save(const std::string &dir_path, std::function<void()> cb, bool text_mode) {
    PullMeta([this, dir_path, cb, text_mode](SparseTensorMeta meta) {
        std::string meta_path = GetSparseMetaPath(dir_path);
        std::string str = meta.ToJsonString();
        EnsureLocalDirectory(dir_path);
        StreamWriteAll(meta_path, str);
        PSMessage req = std::make_shared<Message>();
        json11::Json json = json11::Json::object{
            {"command", "SparseSave"},
            {"name", GetMeta().GetName()},
            {"dir_path", dir_path},
            {"text_mode", text_mode},
        };
        req->GetMessageMeta().SetReceiver(ServerGroup);
        req->GetMessageMeta().SetBody(json.dump());
        agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
    });
}

void SparseTensor::Export(const std::string &dir_path, std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparseExport"},
        {"name", GetMeta().GetName()},
        {"dir_path", dir_path},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void SparseTensor::ImportFrom(const std::string &meta_file_path, std::function<void()> cb,
                              bool data_only, bool skip_existing, bool transform_key,
                              const std::string &feature_name) {
    std::string str = StreamReadAll(meta_file_path);
    SparseTensorMeta meta = SparseTensorMeta::FromJsonString(str);
    if (!meta.IsCompatibleRelaxed(meta_, data_only)) {
        std::string serr;
        serr.append("Incompatible meta detected in '");
        serr.append(meta_file_path);
        serr.append("', can not import sparse tensor '");
        serr.append(GetMeta().GetName());
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    struct Lambda : std::enable_shared_from_this<Lambda> {
        std::string meta_file_path;
        std::function<void()> callback;
        bool data_only = false;
        bool skip_existing = false;
        bool transform_key = false;
        std::string feature_name;
        SparseTensorMeta meta;
        SparseTensor *sparse_tensor = nullptr;
        std::vector<int> partition_indices;
        size_t index = 0;

        void operator()() {
            if (index >= partition_indices.size()) {
                callback();
                return;
            }
            const size_t vec_length =
                data_only ? meta.GetSliceDataLength() : meta.GetSliceTotalBytes();
            ArrayHashMap<uint64_t, uint8_t> map(vec_length);
            const std::string dir_path = DirName(meta_file_path);
            const int partition_index = partition_indices.at(index++);
            std::string path = GetSparsePath(dir_path, meta, partition_index);
            auto stream = Stream::Create(path.c_str(), "r", true);
            if (!stream) {
                std::string serr;
                serr.append("Fail to load partition ");
                serr.append(std::to_string(partition_index));
                serr.append(" of sparse tensor '");
                serr.append(meta.GetName());
                serr.append("' from '");
                serr.append(path);
                serr.append("'.\n\n");
                serr.append(GetStackTrace());
                spdlog::error(serr);
                throw std::runtime_error(serr);
            }
            std::unique_ptr<Stream> stream_guard(stream);
            ArrayHashMapReader reader(meta, map, stream, data_only, transform_key, feature_name,
                                      path);
            MapFileHeader header;
            if (reader.DetectBinaryMode(header)) {
                uint64_t offset = sizeof(header);
                map.deserialize_with_header(
                    path,
                    [stream, &offset](void *ptr, size_t size, const std::string &hint,
                                      const std::string &what) {
                        const size_t nread = stream->Read(ptr, size);
                        if (nread != size) {
                            std::string serr;
                            serr.append(hint);
                            serr.append("incomplete ");
                            serr.append(what);
                            serr.append(", ");
                            serr.append(std::to_string(size));
                            serr.append(" bytes expected, but only ");
                            serr.append(std::to_string(nread));
                            serr.append(" are read successfully. offset = ");
                            serr.append(std::to_string(offset));
                            serr.append("\n\n");
                            serr.append(GetStackTrace());
                            spdlog::error(serr);
                            throw std::runtime_error(serr);
                        }
                    },
                    header);
            } else {
                if (transform_key && feature_name.empty()) {
                    std::string serr;
                    serr.append("Feature name must be specified to transform key; ");
                    serr.append("can not import sparse tensor partition from \"");
                    serr.append(meta_file_path);
                    serr.append("\".\n\n");
                    serr.append(GetStackTrace());
                    spdlog::error(serr);
                    throw std::runtime_error(serr);
                }
                reader.Read();
            }
            sparse_tensor->PushPartition(
                map, [self = shared_from_this()] { (*self)(); }, data_only, skip_existing);
        }
    };
    auto lambda = std::make_shared<Lambda>();
    lambda->meta_file_path = meta_file_path;
    lambda->callback = cb;
    lambda->data_only = data_only;
    lambda->skip_existing = skip_existing;
    lambda->transform_key = transform_key;
    lambda->feature_name = feature_name;
    for (int i = 0; i < meta.GetPartitionCount(); i++)
        if (i % agent_->GetWorkerCount() == agent_->GetAgentRank())
            lambda->partition_indices.push_back(i);
    lambda->meta = std::move(meta);
    lambda->sparse_tensor = this;
    (*lambda)();
}

void SparseTensor::PruneSmall(double epsilon, std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparsePruneSmall"},
        {"name", GetMeta().GetName()},
        {"epsilon", epsilon},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

void SparseTensor::PruneOld(int max_age, std::function<void()> cb) {
    PSMessage req = std::make_shared<Message>();
    json11::Json json = json11::Json::object{
        {"command", "SparsePruneOld"},
        {"name", GetMeta().GetName()},
        {"max_age", max_age},
    };
    req->GetMessageMeta().SetReceiver(ServerGroup);
    req->GetMessageMeta().SetBody(json.dump());
    agent_->BroadcastRequest(req, [cb](PSMessage req, std::vector<PSMessage> ress) { cb(); });
}

std::string SparseTensor::GetSparseMetaPath(const std::string &dir_path) const {
    std::string file_name = fmt::format("{}__sparse_meta.json", GetMeta().GetName());
    std::string file_path = JoinPath(dir_path, file_name);
    return file_path;
}

std::string SparseTensor::GetSparsePath(const std::string &dir_path, const SparseTensorMeta &meta,
                                        int index) {
    std::string file_name = fmt::format("{}__sparse_{}.dat", meta.GetName(), index);
    std::string file_path = JoinPath(dir_path, file_name);
    return file_path;
}

} // namespace metaspore
