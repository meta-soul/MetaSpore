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

#include <math.h>
#include <metaspore/array_hash_map_reader.h>
#include <metaspore/array_hash_map_writer.h>
#include <metaspore/debug.h>
#include <metaspore/io.h>
#include <metaspore/sparse_tensor_partition.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/tensor_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

void SparseTensorPartition::AllocateHashMap() {
    GetMeta().CheckSparseTensorMeta(GetPartitionIndex());
    GetMeta().ComputeSliceInfo();
    Clear();
}

void SparseTensorPartition::Clear() {
    const size_t slice_bytes = GetMeta().GetSliceTotalBytes();
    ArrayHashMap<uint64_t, uint8_t> map(slice_bytes);
    data_.swap(map);
}

void SparseTensorPartition::HandlePush(SmartArray<uint8_t> keys, SmartArray<uint8_t> in,
                                       bool is_value) {
    TransformIndices(keys, false, false);
    const size_t index_count = keys.size() / sizeof(uint64_t);
    const uint64_t *const indices = reinterpret_cast<uint64_t *>(keys.data());
    uint8_t *const param_data = const_cast<uint8_t *>(data_.get_values_array());
    const size_t param_size = GetMeta().GetSliceTotalBytes() * data_.size();
    SparseUpdater updater = GetMeta().GetUpdater();
    if (!updater || is_value) {
        uint8_t *const target_blob = param_data;
        const uint8_t *source = in.data();
        for (size_t i = 0; i < index_count; i++) {
            const uint64_t index = indices[i];
            uint8_t *const target = target_blob + GetMeta().GetSliceTotalBytes() * index;
            memcpy(target, source, GetMeta().GetSliceDataLength());
            source += GetMeta().GetSliceDataLength();
            int &age = *reinterpret_cast<int *>(target + GetMeta().GetSliceAgeOffset());
            age = 0;
        }
    } else {
        uint8_t *const all_keys_data =
            reinterpret_cast<uint8_t *>(const_cast<uint64_t *>(data_.get_keys_array()));
        const size_t all_keys_size = sizeof(uint64_t) * data_.size();
        auto all_keys = SmartArray<uint8_t>::Ref(all_keys_data, all_keys_size);
        auto param = SmartArray<uint8_t>::Ref(param_data, param_size);
        updater(GetMeta().GetName(), param, in, keys, all_keys, GetMeta());
        uint8_t *const target_blob = param_data;
        for (size_t i = 0; i < index_count; i++) {
            const uint64_t index = indices[i];
            uint8_t *const target = target_blob + GetMeta().GetSliceTotalBytes() * index;
            int &age = *reinterpret_cast<int *>(target + GetMeta().GetSliceAgeOffset());
            age = 0;
        }
    }
}

SmartArray<uint8_t> SparseTensorPartition::HandlePull(SmartArray<uint8_t> keys, bool read_only,
                                                      bool nan_fill) {
    TransformIndices(keys, true, read_only);
    const size_t index_count = keys.size() / sizeof(uint64_t);
    const uint64_t *const indices = reinterpret_cast<uint64_t *>(keys.data());
    SmartArray<uint8_t> out(GetMeta().GetSliceDataLength() * index_count);
    uint8_t *target = out.data();
    const uint8_t *const source_blob = data_.get_values_array();
    for (size_t i = 0; i < index_count; i++) {
        const uint64_t index = indices[i];
        const uint8_t *const source = source_blob + GetMeta().GetSliceTotalBytes() * index;
        if (index == kNotFoundIndex && nan_fill)
            FillNaN(target, GetMeta().GetSliceDataLength(), GetMeta().GetDataType());
        else if (index == kNotFoundIndex || index == kPaddingIndex)
            memset(target, 0, GetMeta().GetSliceDataLength());
        else
            memcpy(target, source, GetMeta().GetSliceDataLength());
        target += GetMeta().GetSliceDataLength();
    }
    return std::move(out);
}

void SparseTensorPartition::TransformIndices(SmartArray<uint8_t> keys, bool pull, bool read_only) {
    const size_t index_count = keys.size() / sizeof(uint64_t);
    uint64_t *const indices = reinterpret_cast<uint64_t *>(keys.data());
    if (read_only) {
        for (size_t i = 0; i < index_count; i++)
            if (indices[i] == kPaddingKey)
                indices[i] = kPaddingIndex;
            else
                indices[i] = data_.find(indices[i]);
    } else {
        const size_t old_size = data_.size();
        if (pull) {
            for (size_t i = 0; i < index_count; i++)
                if (indices[i] == kPaddingKey)
                    indices[i] = kPaddingIndex;
                else
                    indices[i] = data_.find_or_init(indices[i]);
        } else {
            for (size_t i = 0; i < index_count; i++)
                indices[i] = data_.find_or_init(indices[i]);
        }
        if (data_.size() != old_size) {
            uint8_t *const values = const_cast<uint8_t *>(data_.get_values_array());
            uint8_t *const blob_data = values + GetMeta().GetSliceTotalBytes() * old_size;
            const size_t blob_size = GetMeta().GetSliceTotalBytes() * (data_.size() - old_size);
            SparseInitializer initializer = GetMeta().GetInitializer();
            if (!initializer)
                memset(blob_data, 0, blob_size);
            else {
                uint8_t *const all_keys =
                    reinterpret_cast<uint8_t *>(const_cast<uint64_t *>(data_.get_keys_array()));
                uint8_t *const blob_keys_data = all_keys + sizeof(uint64_t) * old_size;
                const size_t blob_keys_size = sizeof(uint64_t) * (data_.size() - old_size);
                auto blob = SmartArray<uint8_t>::Ref(blob_data, blob_size);
                auto blob_keys = SmartArray<uint8_t>::Ref(blob_keys_data, blob_keys_size);
                initializer(GetMeta().GetName(), blob, blob_keys, GetMeta());
            }
        }
    }
}

void SparseTensorPartition::HandlePushPartition(SmartArray<uint8_t> keys, SmartArray<uint8_t> in,
                                                bool data_only, bool skip_existing) {
    const size_t vec_length =
        data_only ? GetMeta().GetSliceDataLength() : GetMeta().GetSliceTotalBytes();
    const size_t index_count = keys.size() / sizeof(uint64_t);
    const uint64_t *const indices = reinterpret_cast<uint64_t *>(keys.data());
    const uint8_t *source = in.data();
    for (size_t i = 0; i < index_count; i++) {
        const uint64_t key = indices[i];
        bool is_new;
        uint8_t *target = data_.get_or_init(key, is_new);
        if (is_new || !skip_existing) {
            memcpy(target, source, vec_length);
            if (data_only) {
                // When only the data part of the embedding vector is imported,
                // we set the rest of the embedding vector (state part and age)
                // to zeros. We do this to make it consistent with the behavior
                // of initializers.
                memset(target + GetMeta().GetSliceDataLength(), 0,
                       GetMeta().GetSliceTotalBytes() - GetMeta().GetSliceDataLength());
            }
        }
        source += vec_length;
    }
}

SmartArray<uint8_t> SparseTensorPartition::HandlePullPartition(bool data_only, int index, int count,
                                                               SmartArray<uint8_t> &keys) {
    std::vector<uint64_t> indices;
    std::vector<uint8_t> values;
    const size_t index_count = data_.size();
    const size_t approx_output_count = index_count / count * 2;
    const size_t vec_length =
        data_only ? GetMeta().GetSliceDataLength() : GetMeta().GetSliceTotalBytes();
    indices.reserve(approx_output_count);
    values.reserve(approx_output_count * vec_length);
    const uint64_t *const keys_array = data_.get_keys_array();
    const uint8_t *const values_array = data_.get_values_array();
    for (size_t i = 0; i < index_count; i++) {
        const uint64_t key = indices[i];
        if (key % count == index) {
            indices.push_back(key);
            const uint8_t *const source = values_array + GetMeta().GetSliceTotalBytes() * i;
            VectorAppend(values, source, vec_length);
        }
    }
    auto indices_out = SmartArray<uint64_t>::Wrap(std::move(indices));
    auto values_out = SmartArray<uint8_t>::Wrap(std::move(values));
    keys = indices_out.Cast<uint8_t>();
    return values_out;
}

void SparseTensorPartition::HandlePushMeta(const SparseTensorMeta &meta) {
    if (!meta.IsCompatible(meta_)) {
        std::string serr;
        serr.append("Incompatible meta detected, can not update initializer and updater");
        serr.append(" of sparse tensor '");
        serr.append(GetMeta().GetName());
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    meta_.SetInitializerByData(meta.GetInitializerAsData());
    meta_.SetUpdaterByData(meta.GetUpdaterAsData());
}

const SparseTensorMeta &SparseTensorPartition::HandlePullMeta() { return meta_; }

void SparseTensorPartition::Load(const std::string &dir_path) {
    std::string path = GetSparsePath(dir_path);
    auto stream = Stream::Create(path.c_str(), "r", true);
    if (!stream) {
        std::string serr;
        serr.append("Fail to load partition ");
        serr.append(std::to_string(GetPartitionIndex()));
        serr.append(" of sparse tensor '");
        serr.append(GetMeta().GetName());
        serr.append("' from '");
        serr.append(path);
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    std::unique_ptr<Stream> stream_guard(stream);
    ArrayHashMapReader reader(GetMeta(), data_, stream, false, false, "", path);
    MapFileHeader header;
    if (reader.DetectBinaryMode(header)) {
        uint64_t offset = sizeof(header);
        data_.deserialize_with_header(
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
        reader.Read();
    }
}

void SparseTensorPartition::Save(const std::string &dir_path, bool text_mode) {
    std::string path = GetSparsePath(dir_path);
    auto stream = Stream::Create(path.c_str(), "w", true);
    if (!stream) {
        std::string serr;
        serr.append("Fail to save partition ");
        serr.append(std::to_string(GetPartitionIndex()));
        serr.append(" of sparse tensor '");
        serr.append(GetMeta().GetName());
        serr.append("' to '");
        serr.append(path);
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    std::unique_ptr<Stream> stream_guard(stream);
    if (text_mode) {
        ArrayHashMapWriter writer(GetMeta(), data_);
        writer.Write([stream](const char *ptr, size_t size) { stream->Write(ptr, size); });
    } else {
        const size_t slice_bytes = GetMeta().GetSliceTotalBytes();
        data_.serialize(
            path,
            [stream](const void *ptr, size_t size) {
                // ArrayHashMap can be huge (several gigabytes), writing directly
                // may cause the executor memory to increase substantially, so we
                // write the data block by block. A better solution would be refining
                // the implementation of WriteBuffer::Write.
                const size_t MaxBlockSize = 5 * 1024 * 1024;
                const char *buffer = static_cast<const char *>(ptr);
                while (size > 0) {
                    size_t n = MaxBlockSize;
                    if (n > size)
                        n = size;
                    stream->Write(buffer, n);
                    buffer += n;
                    size -= n;
                }
            },
            slice_bytes);
    }
}

void SparseTensorPartition::Export(const std::string &dir_path) {
    std::string path = GetSparseExportPath(dir_path);
    auto stream = Stream::Create(path.c_str(), "w", true);
    if (!stream) {
        std::string serr;
        serr.append("Fail to export partition ");
        serr.append(std::to_string(GetPartitionIndex()));
        serr.append(" of sparse tensor '");
        serr.append(GetMeta().GetName());
        serr.append("' to '");
        serr.append(path);
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    std::unique_ptr<Stream> stream_guard(stream);
    const size_t data_length = GetMeta().GetSliceDataLength();
    data_.serialize(
        path, [stream](const void *ptr, size_t size) { stream->Write(ptr, size); }, data_length);
}

template <typename T> void SparseTensorPartition::DoPruneSmall(double epsilon) {
    // Only the data part is considered.
    const size_t m = GetMeta().GetSliceDataLength() / sizeof(T);
    data_.prune(
        [epsilon, m, this](uint64_t i, int64_t key, const uint8_t *values, uint64_t value_count) {
            const T *const param = reinterpret_cast<const T *>(values);
            for (uint64_t k = 0; k < m; k++)
                if (fabs(param[k]) > epsilon)
                    return false;
            return true;
        });
}

void SparseTensorPartition::PruneSmall(double epsilon) {
    if (GetMeta().GetDataType() != DataType::Float32 &&
        GetMeta().GetDataType() != DataType::Float64) {
        std::string serr;
        serr.append("SparseTensorPartition::PruneSmall only supports ");
        serr.append("sparse tensors of 'float32' and 'float64'; ");
        serr.append("the data type of sparse tensor '");
        serr.append(GetMeta().GetName());
        serr.append("' is '");
        serr.append(DataTypeToString(GetMeta().GetDataType()));
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (GetMeta().GetDataType() == DataType::Float32)
        DoPruneSmall<float>(epsilon);
    else
        DoPruneSmall<double>(epsilon);
}

void SparseTensorPartition::PruneOld(int max_age) {
    data_.prune(
        [max_age, this](uint64_t i, int64_t key, const uint8_t *values, uint64_t value_count) {
            uint8_t *const ptr = const_cast<uint8_t *>(values);
            int &age = *reinterpret_cast<int *>(ptr + GetMeta().GetSliceAgeOffset());
            ++age;
            return age > max_age;
        });
}

std::string SparseTensorPartition::GetSparsePath(const std::string &dir_path) const {
    std::string file_name =
        fmt::format("{}__sparse_{}.dat", GetMeta().GetName(), GetPartitionIndex());
    std::string file_path = JoinPath(dir_path, file_name);
    return file_path;
}

std::string SparseTensorPartition::GetSparseExportPath(const std::string &dir_path) const {
    std::string file_name =
        fmt::format("part_{}_{}.dat", GetMeta().GetPartitionCount(), GetPartitionIndex());
    std::string file_path = JoinPath(dir_path, file_name);
    return file_path;
}

} // namespace metaspore
