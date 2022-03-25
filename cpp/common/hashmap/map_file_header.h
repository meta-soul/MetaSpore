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

#include <common/hashmap/data_types.h>
#include <stdint.h>
#include <string>

namespace metaspore {

const uint64_t map_file_signature_size = 32;

struct MapFileHeader {
    char signature[map_file_signature_size];
    uint64_t version;
    uint64_t is_optimized_mode;
    uint64_t key_type;
    uint64_t value_type;
    uint64_t key_count;
    uint64_t bucket_count;
    uint64_t value_count;
    uint64_t value_count_per_key;

    void fill_basic_fields(bool optimized_mode);
    bool IsSignatureValid() const;
    void validate(bool optimized_mode, const std::string &hint) const;
};

const uint64_t map_file_header_size = sizeof(MapFileHeader);

extern const char map_file_signature[map_file_signature_size];
extern const uint64_t map_file_version;

} // namespace metaspore