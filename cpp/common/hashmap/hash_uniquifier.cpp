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

#include <common/hashmap/hash_uniquifier.h>
#include <common/hashmap/hashtable_helpers.h>
#include <utility>

namespace metaspore {

std::vector<uint64_t> HashUniquifier::Uniquify(uint64_t *items, size_t count) {
    const uint64_t capacity = GetHashCapacity(count);
    const uint64_t size = capacity * 2 / 3;
    std::vector<uint64_t> entries;
    std::vector<int32_t> buckets;
    entries.reserve(size);
    buckets.assign(capacity, -1);
    for (size_t i = 0; i < count; i++) {
        uint64_t offset;
        const uint64_t key = items[i];
        InsertHashEntry(key, offset, entries, buckets);
        items[i] = offset;
    }
    return std::move(entries);
}

std::vector<uint64_t> HashUniquifier::Uniquify(std::vector<uint64_t> &items) {
    return Uniquify(items.data(), items.size());
}

int32_t HashUniquifier::FindEntryAndBucket(uint64_t key, uint64_t hashCode,
                                           const std::vector<uint64_t> &entries,
                                           const std::vector<int32_t> &buckets, uint64_t &bucket) {
    const uint64_t mask = buckets.size() - 1;
    uint64_t perturb = hashCode;
    bucket = hashCode & mask;
    for (;;) {
        const int32_t i = buckets.at(bucket);
        if (i == -1)
            return -1;
        if (i >= 0 && entries.at(i) == key)
            return i;
        perturb >>= 5;
        bucket = (bucket * 5 + 1 + perturb) & mask;
    }
}

uint64_t HashUniquifier::GetHashCapacity(uint64_t minSize) {
    const uint64_t cap = (3 * minSize + 1) / 2;
    return HashtableHelpers::get_power_bucket_count(cap);
}

bool HashUniquifier::InsertHashEntry(uint64_t key, uint64_t &offset, std::vector<uint64_t> &entries,
                                     std::vector<int32_t> &buckets) {
    uint64_t bucket;
    const uint64_t hashCode = HashtableHelpers::FastModulo(key);
    const int32_t n = FindEntryAndBucket(key, hashCode, entries, buckets, bucket);
    if (n == -1) {
        offset = static_cast<uint64_t>(entries.size());
        buckets.at(bucket) = static_cast<int32_t>(offset);
        entries.push_back(key);
        return true;
    }
    offset = static_cast<uint64_t>(n);
    return false;
}

} // namespace metaspore
