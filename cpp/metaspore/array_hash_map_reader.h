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

#include <common/hash_utils.h>
#include <common/hashmap/array_hash_map.h>
#include <metaspore/io.h>
#include <metaspore/sparse_tensor_meta.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/string_utils.h>
#include <sstream>

namespace metaspore {

class ArrayHashMapReader {
  public:
    ArrayHashMapReader(SparseTensorMeta &meta, ArrayHashMap<uint64_t, uint8_t> &data,
                       Stream *stream, bool data_only, bool transform_key, std::string feature_name,
                       const std::string &path)
        : meta_(meta), data_(data), stream_(stream), data_only_(data_only),
          transform_key_(transform_key), feature_name_(feature_name),
          feature_name_hash_(BKDRHashWithEqualPostfix(feature_name)), path_(path) {}

    bool DetectBinaryMode(MapFileHeader &header) {
        void *const ptr = static_cast<void *>(&header);
        const size_t size = sizeof(header);
        const size_t nread = stream_->Read(ptr, size);
        if (nread == size && header.IsSignatureValid())
            return true;
        buffer_.append(static_cast<char *>(ptr), nread);
        return false;
    }

    void Read() {
        size_t lineno = 0;
        while (FillBuffer()) {
            size_t pos = 0;
            size_t n = buffer_.find(line_terminator, pos);
            while (n != std::string::npos) {
                std::string_view text{buffer_.data() + pos, n - pos};
                ProcessLine(++lineno, text);
                pos = n + line_terminator.size();
                n = buffer_.find(line_terminator, pos);
            }
            buffer_.erase(buffer_.begin(), buffer_.begin() + pos);
        }
    }

  private:
    static constexpr size_t buffer_size = 2 * 1024 * 1024;
    static constexpr std::string_view key_value_separator = "\t";
    static constexpr std::string_view field_separator = "|";
    static constexpr std::string_view value_separator = ",";
    static constexpr std::string_view line_terminator = "\n";

    bool FillBuffer() {
        if (eof_reached_)
            return false;
        read_buffer_.resize(buffer_size);
        const size_t nread = stream_->Read(read_buffer_.data(), read_buffer_.size());
        buffer_.append(read_buffer_.data(), nread);
        if (nread == 0) {
            if (!buffer_.empty() && !EndsWith(buffer_, line_terminator))
                buffer_.append(line_terminator);
            eof_reached_ = true;
        }
        return !buffer_.empty();
    }

    void ProcessLine(size_t lineno, std::string_view text) {
        auto key_and_value = SplitStringView(text, key_value_separator);
        if (key_and_value.size() != 2) {
            std::string serr;
            serr.append("Line ");
            serr.append(std::to_string(lineno));
            serr.append(" of file \"");
            serr.append(path_);
            serr.append("\" is not valid key-value pair separated by ");
            serr.append(ToSource(key_value_separator));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        const uint64_t key = ParseKey(lineno, key_and_value.at(0));
        uint8_t *values = data_.get_or_init(key);
        ParseValues(lineno, key_and_value.at(1), values);
    }

    uint64_t ParseKey(size_t lineno, std::string_view str) const {
        if (transform_key_)
            return ParseStringKey(str);
        else
            return ParseUInt64Key(lineno, str);
    }

    uint64_t ParseStringKey(std::string_view str) const {
        const uint64_t name = feature_name_hash_;
        const uint64_t value = BKDRHash(str);
        const uint64_t key = CombineHashCodes(name, value);
        return key;
    }

    uint64_t ParseUInt64Key(size_t lineno, std::string_view str) const {
        try {
            const uint64_t key = std::stoul(std::string(str));
            return key;
        } catch (const std::exception &e) {
            std::string serr;
            serr.append("Key ");
            serr.append(ToSource(str));
            serr.append("of line ");
            serr.append(std::to_string(lineno));
            serr.append(" of file \"");
            serr.append(path_);
            serr.append("\" can not be parsed as uint64_t. ");
            serr.append(e.what());
            serr.append("\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }

    void ParseValues(size_t lineno, std::string_view str, uint8_t *values) {
        const DataType type = meta_.GetDataType();
        switch (type) {
#undef MS_DATA_TYPE_DEF
#define MS_DATA_TYPE_DEF(t, l, u)                                                                  \
    case DataType::u:                                                                              \
        ParseValuesTyped<t>(lineno, str, values);                                                  \
        break;
            MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_TYPE_DEF)
        default:
            std::string serr;
            serr.append("Invalid DataType enum value: ");
            serr.append(std::to_string(static_cast<int>(type)));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }

    template <typename TValue>
    void ParseValuesTyped(size_t lineno, std::string_view str, uint8_t *values) {
        const size_t data_count = meta_.GetSliceDataLength() / sizeof(TValue);
        TValue *const data_items = reinterpret_cast<TValue *>(values);
        if (data_only_)
            ParseDataOrStateValues(lineno, str, data_count, data_items, true);
        else {
            auto fields = SplitStringView(str, field_separator);
            if (fields.size() != 3) {
                std::string serr;
                serr.append("Fail to parse fields separated by ");
                serr.append(ToSource(field_separator));
                serr.append(" at line ");
                serr.append(std::to_string(lineno));
                serr.append(" of file \"");
                serr.append(path_);
                serr.append("\". Expect data");
                serr.append(field_separator);
                serr.append("state");
                serr.append(field_separator);
                serr.append("age 3 fields, found ");
                serr.append(std::to_string(fields.size()));
                serr.append(".\n\n");
                serr.append(GetStackTrace());
                spdlog::error(serr);
                throw std::runtime_error(serr);
            }
            ParseDataOrStateValues(lineno, fields.at(0), data_count, data_items, true);
            const size_t state_count = meta_.GetSliceStateLength() / sizeof(TValue);
            TValue *const state_items =
                reinterpret_cast<TValue *>(values + meta_.GetSliceDataLength());
            ParseDataOrStateValues(lineno, fields.at(1), state_count, state_items, false);
            int &age = *reinterpret_cast<int *>(values + meta_.GetSliceAgeOffset());
            ParseAgeValue(lineno, fields.at(2), age);
        }
    }

    template <typename TValue>
    void ParseDataOrStateValues(size_t lineno, std::string_view text, size_t count, TValue *items,
                                bool is_data) {
        auto strs = SplitStringView(text, value_separator);
        if (strs.size() != count) {
            std::string serr;
            serr.append("Fail to parse ");
            serr.append(is_data ? "data" : "state");
            serr.append(" values separated by ");
            serr.append(ToSource(value_separator));
            serr.append(" at line ");
            serr.append(std::to_string(lineno));
            serr.append(" of file \"");
            serr.append(path_);
            serr.append("\". Expect ");
            serr.append(std::to_string(count));
            serr.append(" values, found ");
            serr.append(std::to_string(strs.size()));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        for (size_t i = 0; i < count; i++) {
            TValue value;
            std::istringstream sin(std::string{strs.at(i)});
            if (sin >> value)
                items[i] = value;
            else {
                std::string serr;
                serr.append("Fail to parse ");
                serr.append(is_data ? "data" : "state");
                serr.append(" value ");
                serr.append(std::to_string(i));
                serr.append(" at line ");
                serr.append(std::to_string(lineno));
                serr.append(" of file \"");
                serr.append(path_);
                serr.append("\".\n\n");
                serr.append(GetStackTrace());
                spdlog::error(serr);
                throw std::runtime_error(serr);
            }
        }
    }

    void ParseAgeValue(size_t lineno, std::string_view text, int &age) {
        int value;
        std::istringstream sin(std::string{text});
        if (sin >> value)
            age = value;
        else {
            std::string serr;
            serr.append("Fail to parse age value at line ");
            serr.append(std::to_string(lineno));
            serr.append(" of file \"");
            serr.append(path_);
            serr.append("\".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
    }

    SparseTensorMeta &meta_;
    ArrayHashMap<uint64_t, uint8_t> &data_;
    Stream *stream_;
    bool data_only_;
    bool transform_key_;
    std::string feature_name_;
    uint64_t feature_name_hash_;
    std::string path_;
    std::string buffer_;
    std::string read_buffer_;
    bool eof_reached_ = false;
};

} // namespace metaspore
