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

#include <common/hashmap/map_file_header.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <utility>
#include <vector>

namespace metaspore {

template <typename TKey, typename TValue> class MemoryMappedArrayHashMap {
  public:
    explicit MemoryMappedArrayHashMap(std::shared_ptr<void> blob) {
        const void *ptr = blob.get();
        const MapFileHeader &header = *static_cast<const MapFileHeader *>(ptr);

        uint64_t value_count = header.value_count;
        uint64_t value_count_per_key = header.value_count_per_key;

        if (header.key_type != static_cast<uint64_t>(DataTypeToCode<TKey>::value)) {
            const DataType key_type_1 = static_cast<DataType>(header.key_type);
            const DataType key_type_2 = DataTypeToCode<TKey>::value;
            const size_t key_size_1 = DataTypeToSize(key_type_1);
            const size_t key_size_2 = DataTypeToSize(key_type_2);
            if (key_size_1 != key_size_2) {
                std::ostringstream serr;
                serr << "key types mismatch; ";
                serr << "expect '" << DataTypeToString(DataTypeToCode<TKey>::value) << "', ";
                serr << "found '" << DataTypeToString(static_cast<DataType>(header.key_type))
                     << "'.";
                throw std::runtime_error(serr.str());
            }
        }
        if (header.value_type != static_cast<uint64_t>(DataTypeToCode<TValue>::value)) {
            const DataType value_type_1 = static_cast<DataType>(header.value_type);
            const DataType value_type_2 = DataTypeToCode<TValue>::value;
            const size_t value_size_1 = DataTypeToSize(value_type_1);
            const size_t value_size_2 = DataTypeToSize(value_type_2);
            if (value_size_1 != value_size_2) {
                if (value_count_per_key * value_size_1 % value_size_2 == 0) {
                    value_count = value_count * value_size_1 / value_size_2;
                    value_count_per_key = value_count_per_key * value_size_1 / value_size_2;
                } else {
                    std::ostringstream serr;
                    serr << "value types mismatch; ";
                    serr << "expect '" << DataTypeToString(DataTypeToCode<TValue>::value) << "', ";
                    serr << "found '" << DataTypeToString(static_cast<DataType>(header.value_type))
                         << "'. ";
                    serr << "value_count_per_key = " << value_count_per_key;
                    throw std::runtime_error(serr.str());
                }
            }
        }

        key_count_ = header.key_count;
        bucket_count_ = header.bucket_count;
        value_count_ = value_count;
        value_count_per_key_ = value_count_per_key;

        const char *address = static_cast<const char *>(ptr);
        address += sizeof(MapFileHeader);
        keys_ = reinterpret_cast<const TKey *>(address);
        address += key_count_ * sizeof(TKey);
        values_ = reinterpret_cast<const TValue *>(address);
        address += value_count_ * sizeof(TValue);
        next_ = reinterpret_cast<const uint32_t *>(address);
        address += key_count_ * sizeof(uint32_t);
        first_ = reinterpret_cast<const uint32_t *>(address);
        address += bucket_count_ * sizeof(uint32_t);

        blob_ = std::move(blob);
    }

    uint64_t get_value_count_per_key() const { return value_count_per_key_; }

    uint64_t size() const { return key_count_; }

    bool empty() const { return key_count_ == 0; }

    bool contains(TKey key) const {
        if (bucket_count_ == 0)
            return false;
        const uint32_t nil = uint32_t(-1);
        const uint64_t bucket = get_bucket(key);
        uint32_t i = first_[bucket];
        while (i != nil) {
            if (keys_[i] == key)
                return true;
            i = next_[i];
        }
        return false;
    }

    int64_t find(TKey key) const {
        if (bucket_count_ == 0)
            return -1;
        const uint32_t nil = uint32_t(-1);
        const uint64_t bucket = get_bucket(key);
        uint32_t i = first_[bucket];
        while (i != nil) {
            if (keys_[i] == key)
                return static_cast<int64_t>(i);
            i = next_[i];
        }
        return -1;
    }

    const TValue *get(TKey key) const {
        if (bucket_count_ == 0)
            return nullptr;
        const uint32_t nil = uint32_t(-1);
        const uint64_t bucket = get_bucket(key);
        uint32_t i = first_[bucket];
        while (i != nil) {
            if (keys_[i] == key)
                return &values_[i * value_count_per_key_];
            i = next_[i];
        }
        return nullptr;
    }

    const TValue *get(TKey key, uint64_t &value_count) const {
        if (bucket_count_ == 0) {
            value_count = 0;
            return nullptr;
        }
        const uint32_t nil = uint32_t(-1);
        const uint64_t bucket = get_bucket(key);
        uint32_t i = first_[bucket];
        while (i != nil) {
            if (keys_[i] == key) {
                value_count = value_count_per_key_;
                return &values_[i * value_count_per_key_];
            }
            i = next_[i];
        }
        value_count = 0;
        return nullptr;
    }

    void dump(std::ostream &out = std::cerr, uint64_t count_limit = uint64_t(-1)) const {
        for (uint64_t i = 0; i < key_count_; i++) {
            if (i >= count_limit)
                break;
            const TKey key = keys_[i];
            const uint64_t value_count = value_count_per_key_;
            const TValue *const values = &values_[i * value_count_per_key_];
            out << key << ": [";
            for (uint64_t j = 0; j < value_count; j++)
                out << (j ? ", " : "") << as_number(values[j]);
            out << "]\n";
        }
    }

    class iterator {
      public:
        iterator(const MemoryMappedArrayHashMap<TKey, TValue> *map, uint64_t index)
            : map_(map), index_(index) {}

        iterator &operator++() {
            if (index_ < map_->key_count_)
                index_++;
            return *this;
        }

        TKey operator*() const {
            if (index_ < map_->key_count_)
                return map_->keys_[index_];
            else
                return TKey(-1);
        }

        bool operator==(const iterator &rhs) const {
            return index_ == rhs.index_ && map_ == rhs.map_;
        }

        bool operator!=(const iterator &rhs) const { return !(*this == rhs); }

      private:
        const MemoryMappedArrayHashMap<TKey, TValue> *map_;
        uint64_t index_;
    };

    iterator begin() const { return iterator(this, 0); }

    iterator end() const { return iterator(this, key_count_); }

    template <typename Func> void each(Func action) {
        for (uint64_t i = 0; i < key_count_; i++) {
            const TKey key = keys_[i];
            const uint64_t value_count = value_count_per_key_;
            const TValue *const values = &values_[i * value_count_per_key_];
            action(i, key, values, value_count);
        }
    }

    uint64_t get_hash_code() const {
        uint64_t hash = 0;
        for (uint64_t key : *this) {
            uint64_t c = key;
            uint64_t value_count;
            const TValue *const values = get(key, value_count);
            const uint8_t *const bytes = reinterpret_cast<const uint8_t *>(values);
            const uint64_t num_bytes = value_count * sizeof(TValue);
            for (uint64_t i = 0; i < num_bytes; i++)
                c = c * 31 + bytes[i];
            hash ^= c;
        }
        return hash;
    }

    void show_statistics(std::ostream &sout = std::cerr) const {
        std::vector<uint64_t> counter(bucket_count_);
        for (uint64_t i = 0; i < key_count_; i++) {
            const TKey key = keys_[i];
            const uint64_t bucket = get_bucket(key);
            counter.at(bucket)++;
        }
        std::vector<uint64_t> lens;
        for (uint64_t len : counter) {
            while (lens.size() <= len)
                lens.push_back(0);
            lens.at(len)++;
        }
        for (uint64_t k = 0; k < lens.size(); k++)
            sout << k << ": " << lens.at(k) << std::endl;
    }

  private:
    static constexpr uint64_t fast_modulo(uint64_t a) {
        constexpr int q = 31;
        constexpr uint64_t prime = (UINT64_C(1) << q) - 1;
        uint64_t r = (a & prime) + (a >> q);
        if (r >= prime)
            r -= prime;
        return r;
    }

    uint64_t get_bucket(TKey key) const {
        return fast_modulo(static_cast<uint64_t>(key)) & (bucket_count_ - 1);
    }

    std::shared_ptr<void> blob_;
    uint64_t key_count_;
    uint64_t bucket_count_;
    uint64_t value_count_;
    uint64_t value_count_per_key_;
    const TKey *keys_;
    const TValue *values_;
    const uint32_t *next_;
    const uint32_t *first_;
};

} // namespace metaspore