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

#include <common/hashmap/array_hash_map.h>
#include <cstdio>
#include <metaspore/sparse_tensor_meta.h>
#include <metaspore/stack_trace_utils.h>

namespace metaspore {

class ArrayHashMapWriter {
  public:
    ArrayHashMapWriter(SparseTensorMeta &meta, ArrayHashMap<uint64_t, uint8_t> &data)
        : meta_(meta), data_(data) {}

    template <typename Func> void Write(Func write) {
        const DataType type = meta_.GetDataType();
        switch (type) {
#undef METASPORE_DATA_TYPE_DEF
#define METASPORE_DATA_TYPE_DEF(t, l, u)                                                           \
    case DataType::u:                                                                              \
        WriteData<Func, t>(write);                                                                 \
        break;
            MS_DATA_STRUCTURES_DATA_TYPES(METASPORE_DATA_TYPE_DEF)
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

  private:
    static constexpr size_t buffer_size = 256;
    static constexpr char key_value_separator = '\t';
    static constexpr char field_separator = '|';
    static constexpr char value_separator = ',';
    static constexpr char line_terminator = '\n';

    template <typename Func, typename TValue> void WriteData(Func write) {
        data_.each([this, write](uint64_t i, uint64_t key, const uint8_t *values, uint64_t count) {
            std::string sout;
            char buffer[buffer_size];
            Append(sout, buffer, key);
            sout.push_back(key_value_separator);
            const size_t data_count = meta_.GetSliceDataLength() / sizeof(TValue);
            const TValue *const data_items = reinterpret_cast<const TValue *>(values);
            for (size_t i = 0; i < data_count; i++) {
                if (i > 0)
                    sout.push_back(value_separator);
                Append(sout, buffer, data_items[i]);
            }
            sout.push_back(field_separator);
            const size_t state_count = meta_.GetSliceStateLength() / sizeof(TValue);
            const TValue *const state_items =
                reinterpret_cast<const TValue *>(values + meta_.GetSliceDataLength());
            for (size_t i = 0; i < state_count; i++) {
                if (i > 0)
                    sout.push_back(value_separator);
                Append(sout, buffer, state_items[i]);
            }
            sout.push_back(field_separator);
            const int age = *reinterpret_cast<const int *>(values + meta_.GetSliceAgeOffset());
            Append(sout, buffer, age);
            sout.push_back(line_terminator);
            write(sout.data(), sout.size());
        });
    }

    template <typename T>
    static void Append(std::string &sout, char (&buffer)[buffer_size], T value) {
        sout.append(std::to_string(value));
    }

    static void Append(std::string &sout, char (&buffer)[buffer_size], float value) {
        sout.append(buffer, std::sprintf(buffer, "%.7g", value));
    }

    static void Append(std::string &sout, char (&buffer)[buffer_size], double value) {
        sout.append(buffer, std::sprintf(buffer, "%.15g", value));
    }

    SparseTensorMeta &meta_;
    ArrayHashMap<uint64_t, uint8_t> &data_;
};

} // namespace metaspore
