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

#include "metaspore/debug.h"
#include "metaspore/io.h"
#include "metaspore/smart_array.h"
#include <common/logger.h>

namespace metaspore {

struct FileHeader {
    uint32_t magic;
    uint32_t patch;
    uint64_t size;
    static const uint32_t kMagicNum = 0xffffeeee;
    bool Check() { return magic == kMagicNum && size > 0; }
    inline uint64_t Size() { return size; }
};

template <typename T> int LoadAsSArray(const std::string &path, SmartArray<T> *array) {
    if (array->empty()) {
        // ignore empty range on this server
        SPDLOG_INFO("Ignoring empty range for {}", path);
        return 0;
    }
    std::unique_ptr<Stream> stream(Stream::Create(path.c_str(), "r", true));
    if (!stream) {
        return -1;
    }
    const size_t nread = stream->Read(array->data(), array->size());
    if (nread != array->size())
        return -1;
    return 0;
}

template <typename T> int SaveAsSArray(const std::string &path, const SmartArray<T> &array) {
    if (array.empty()) {
        // ignore empty range on this server
        SPDLOG_INFO("Ignoring empty range for {}", path);
        return 0;
    }
    std::unique_ptr<Stream> stream(Stream::Create(path.c_str(), "w", true));
    if (!stream) {
        return -1;
    }
    stream->Write(array.data(), array.size());
    return 0;
}

} // namespace metaspore
