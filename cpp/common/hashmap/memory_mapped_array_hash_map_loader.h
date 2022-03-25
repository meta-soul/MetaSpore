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
#include <common/hashmap/hashtable_helpers.h>
#include <common/hashmap/map_file_header.h>
#include <common/hashmap/memory_mapped_array_hash_map.h>
#include <common/hashmap/perfect_array_hash_map.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string.h>
#include <string>

namespace metaspore {

class MemoryMappedArrayHashMapLoader {
  public:
    explicit MemoryMappedArrayHashMapLoader(const std::string &path, bool disableMmap = false);

    const MapFileHeader &get_header() const {
        const void *ptr = blob_.get();
        const MapFileHeader &header = *static_cast<const MapFileHeader *>(ptr);
        return header;
    }

    template <typename TKey, typename TValue> MemoryMappedArrayHashMap<TKey, TValue> get() const {
        if (is_perfect_hashmap() || get_header().bucket_count <= 2 ||
            HashtableHelpers::get_prime_bucket_count(get_header().bucket_count) ==
                get_header().bucket_count) {
            convert_blob<TKey, TValue>(false);
        }
        return MemoryMappedArrayHashMap<TKey, TValue>(blob_);
    }

    template <typename TKey, typename TValue>
    PerfectArrayHashMap<TKey, TValue> get_optimized() const {
        if (!is_perfect_hashmap()) {
            convert_blob<TKey, TValue>(true);
        }
        return PerfectArrayHashMap<TKey, TValue>(blob_);
    }
    bool is_perfect_hashmap() const { return get_header().is_optimized_mode == 1; }

  private:
    template <typename TKey, typename TValue> void convert_blob(bool optimized) const {
        std::shared_ptr<void> old_blob = blob_;
        ArrayHashMap<TKey, TValue> map;
        uint64_t offset = 0;
        map.deserialize(path_, [this, old_blob, &offset](void *ptr, size_t size,
                                                         const std::string &hint,
                                                         const std::string &what) {
            if (offset + size > size_) {
                const size_t nread = size_ - offset;
                std::ostringstream serr;
                serr << hint;
                serr << "incomplete " << what << ", ";
                serr << size << " bytes expected, ";
                serr << "but only " << nread << " are read successfully. ";
                serr << "offset = " << offset << " (0x" << std::hex << offset << ")";
                throw std::runtime_error(serr.str());
            }
            const unsigned char *const begin = static_cast<unsigned char *>(old_blob.get());
            memcpy(ptr, begin + offset, size);
            offset += size;
        });
        auto new_buffer = std::make_shared<std::string>();
        map.serialize(
            path_ + " [memory]",
            [new_buffer](const void *ptr, size_t size) {
                new_buffer->append(static_cast<const char *>(ptr), size);
            },
            static_cast<uint64_t>(-1), optimized);
        blob_ = std::shared_ptr<void>(new_buffer, new_buffer->data());
    }

    std::string path_;
    mutable std::shared_ptr<void> blob_;
    uint64_t size_;
};

} // namespace metaspore