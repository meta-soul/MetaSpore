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

#include <common/hashmap/hashtable_helpers.h>
#include <common/hashmap/map_file_header.h>
#include <common/hashmap/memory_buffer.h>
#include <common/hashmap/perfect_hash_index_builder.h>
#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <utility>
#include <vector>

namespace metaspore {

template <typename TKey, typename TValue> class ArrayHashMap {
  public:
    ArrayHashMap() { static_assert(sizeof(TKey) <= sizeof(uint64_t), "invalid key type"); }

    explicit ArrayHashMap(int64_t value_count_per_key) : ArrayHashMap() {
        if (value_count_per_key < 0)
            throw std::runtime_error("value_count_per_key must be non-negative.");
        value_count_per_key_ = static_cast<uint64_t>(value_count_per_key);
    }

    ArrayHashMap(ArrayHashMap &&rhs)
        : keys_buffer_(std::move(rhs.keys_buffer_)), values_buffer_(std::move(rhs.values_buffer_)),
          next_buffer_(std::move(rhs.next_buffer_)), first_buffer_(std::move(rhs.first_buffer_)),
          key_count_(rhs.key_count_), bucket_count_(rhs.bucket_count_),
          value_count_(rhs.value_count_), value_count_per_key_(rhs.value_count_per_key_),
          keys_(rhs.keys_), values_(rhs.values_), next_(rhs.next_), first_(rhs.first_) {
        rhs.key_count_ = 0;
        rhs.bucket_count_ = 0;
        rhs.value_count_ = 0;
        rhs.value_count_per_key_ = static_cast<uint64_t>(-1);
        rhs.keys_ = nullptr;
        rhs.values_ = nullptr;
        rhs.next_ = nullptr;
        rhs.first_ = nullptr;
    }

    ~ArrayHashMap() {
        key_count_ = 0;
        bucket_count_ = 0;
        value_count_ = 0;
        value_count_per_key_ = static_cast<uint64_t>(-1);
        keys_ = nullptr;
        values_ = nullptr;
        next_ = nullptr;
        first_ = nullptr;
    }

    void swap(ArrayHashMap &other) {
        keys_buffer_.swap(other.keys_buffer_);
        values_buffer_.swap(other.values_buffer_);
        next_buffer_.swap(other.next_buffer_);
        first_buffer_.swap(other.first_buffer_);
        std::swap(key_count_, other.key_count_);
        std::swap(bucket_count_, other.bucket_count_);
        std::swap(value_count_, other.value_count_);
        std::swap(value_count_per_key_, other.value_count_per_key_);
        std::swap(keys_, other.keys_);
        std::swap(values_, other.values_);
        std::swap(next_, other.next_);
        std::swap(first_, other.first_);
    }

    uint64_t get_key_count() const { return key_count_; }
    uint64_t get_bucket_count() const { return bucket_count_; }
    uint64_t get_value_count() const { return value_count_; }
    uint64_t get_value_count_per_key() const { return value_count_per_key_; }

    const TKey *get_keys_array() const { return keys_; }
    const TValue *get_values_array() const { return values_; }
    const uint32_t *get_next_array() const { return next_; }
    const uint32_t *get_first_array() const { return first_; }

    void set_value_count_per_key(int64_t value_count_per_key) {
        if (value_count_per_key < 0)
            throw std::runtime_error("value_count_per_key must be non-negative.");
        if (value_count_per_key_ != static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key has been set.");
        value_count_per_key_ = static_cast<uint64_t>(value_count_per_key);
    }

    void reserve(uint64_t size) {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        if (bucket_count_ >= size)
            return;
        reallocate(size);
    }

    void reallocate(uint64_t size) {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        if (key_count_ > size)
            return;
        if (size == 0) {
            deallocate();
            return;
        }
        const uint64_t bucket_count = HashtableHelpers::get_power_bucket_count(size);
        const uint64_t limit = std::numeric_limits<uint32_t>::max();
        if (bucket_count > limit) {
            std::ostringstream serr;
            serr << "store " << size << " keys ";
            serr << "requires " << bucket_count << " buckets, ";
            serr << "but at most " << limit << " are allowed.";
            throw std::runtime_error(serr.str());
        }
        keys_buffer_.reallocate(bucket_count * sizeof(TKey));
        values_buffer_.reallocate(bucket_count * value_count_per_key_ * sizeof(TValue));
        next_buffer_.reallocate(bucket_count * sizeof(uint32_t));
        first_buffer_.reallocate(bucket_count * sizeof(uint32_t));
        bucket_count_ = bucket_count;
        keys_ = static_cast<TKey *>(keys_buffer_.get_pointer());
        values_ = static_cast<TValue *>(values_buffer_.get_pointer());
        next_ = static_cast<uint32_t *>(next_buffer_.get_pointer());
        first_ = static_cast<uint32_t *>(first_buffer_.get_pointer());
        build_hash_index();
    }

    uint64_t size() const { return key_count_; }

    bool empty() const { return key_count_ == 0; }

    uint64_t get_bucket(TKey key) const {
        return fast_modulo(static_cast<uint64_t>(key)) & (bucket_count_ - 1);
    }

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

    int64_t find_or_init(TKey key) {
        bool is_new;
        int64_t index;
        get_or_init(key, is_new, index);
        return index;
    }

    const TValue *get(TKey key) const {
        if (bucket_count_ == 0)
            return nullptr;
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
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
        const TValue *const ptr = get(key);
        value_count = ptr ? value_count_per_key_ : 0;
        return ptr;
    }

    TValue *get(TKey key) {
        return const_cast<TValue *>(static_cast<const ArrayHashMap *>(this)->get(key));
    }

    TValue *get(TKey key, uint64_t &value_count) {
        return const_cast<TValue *>(static_cast<const ArrayHashMap *>(this)->get(key, value_count));
    }

    TValue *get_or_init(TKey key, bool &is_new, int64_t &index) {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        const uint32_t nil = uint32_t(-1);
        if (bucket_count_ > 0) {
            const uint64_t b = get_bucket(key);
            uint32_t i = first_[b];
            while (i != nil) {
                if (keys_[i] == key) {
                    is_new = false;
                    index = i;
                    return &values_[i * value_count_per_key_];
                }
                i = next_[i];
            }
        }
        if (key_count_ == bucket_count_)
            ensure_capacity();
        const uint64_t bucket = get_bucket(key);
        index = static_cast<int64_t>(key_count_);
        keys_[index] = key;
        next_[index] = first_[bucket];
        first_[bucket] = static_cast<uint32_t>(index);
        is_new = true;
        key_count_++;
        value_count_ += value_count_per_key_;
        return &values_[index * value_count_per_key_];
    }

    TValue *get_or_init(TKey key, bool &is_new) {
        int64_t index;
        return get_or_init(key, is_new, index);
    }

    TValue *get_or_init(TKey key) {
        bool is_new;
        return get_or_init(key, is_new);
    }

    void put(TKey key, const TValue *values) {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        TValue *buffer = get_or_init(key);
        memcpy(buffer, values, value_count_per_key_ * sizeof(TValue));
    }

    void put(TKey key, const std::vector<TValue> &values) {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        if (values.size() != value_count_per_key_) {
            std::ostringstream serr;
            serr << "incorrect number of values; ";
            serr << value_count_per_key_ << " expected, ";
            serr << "but found " << values.size() << ".";
            throw std::runtime_error(serr.str());
        }
        put(key, &values[0]);
    }

    void clear() {
        key_count_ = 0;
        value_count_ = 0;
        build_hash_index();
    }

    void deallocate() {
        keys_buffer_.deallocate();
        values_buffer_.deallocate();
        next_buffer_.deallocate();
        first_buffer_.deallocate();
        key_count_ = 0;
        bucket_count_ = 0;
        value_count_ = 0;
        keys_ = nullptr;
        values_ = nullptr;
        next_ = nullptr;
        first_ = nullptr;
    }

    void reset() {
        deallocate();
        value_count_per_key_ = static_cast<uint64_t>(-1);
    }

    template <typename Func> void prune(Func pred) {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        uint64_t v = 0;
        for (uint64_t i = 0; i < key_count_; i++) {
            const TKey key = keys_[i];
            const TValue *values = &values_[i * value_count_per_key_];
            if (!pred(i, key, values, value_count_per_key_)) {
                if (v != i) {
                    keys_[v] = key;
                    memcpy(&values_[v * value_count_per_key_], values,
                           value_count_per_key_ * sizeof(TValue));
                }
                v++;
            }
        }
        if (v < key_count_) {
            key_count_ = v;
            value_count_ = v * value_count_per_key_;
            reallocate(key_count_);
        }
    }

    void dump(std::ostream &out = std::cerr, uint64_t count_limit = uint64_t(-1)) const {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        for (uint64_t i = 0; i < key_count_; i++) {
            if (i >= count_limit)
                break;
            TKey key = keys_[i];
            ;
            const TValue *values = &values_[i * value_count_per_key_];
            out << key << ": [";
            for (uint64_t j = 0; j < value_count_per_key_; j++)
                out << (j ? ", " : "") << as_number(values[j]);
            out << "]\n";
        }
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

    uint64_t get_hash_code() const {
        uint64_t hash = 0;
        for (uint64_t key : *this) {
            uint64_t c = key;
            const TValue *const values = get(key);
            const uint8_t *const bytes = reinterpret_cast<const uint8_t *>(values);
            const uint64_t num_bytes = value_count_per_key_ * sizeof(TValue);
            for (uint64_t i = 0; i < num_bytes; i++)
                c = c * 31 + bytes[i];
            hash ^= c;
        }
        return hash;
    }

    class iterator {
      public:
        iterator(const ArrayHashMap<TKey, TValue> *map, uint64_t index)
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
        const ArrayHashMap<TKey, TValue> *map_;
        uint64_t index_;
    };

    iterator begin() const { return iterator(this, 0); }

    iterator end() const { return iterator(this, key_count_); }

    template <typename Func> void each(Func action) {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        for (uint64_t i = 0; i < key_count_; i++) {
            const TKey key = keys_[i];
            const TValue *values = &values_[i * value_count_per_key_];
            action(i, key, values, value_count_per_key_);
        }
    }

    template <typename Func>
    int serialize(const std::string &path, Func write,
                  uint64_t value_count_per_key = static_cast<uint64_t>(-1),
                  bool optimized_mode = false) {
        if (value_count_per_key_ == static_cast<uint64_t>(-1))
            throw std::runtime_error("value_count_per_key is not set.");
        if (value_count_per_key == static_cast<uint64_t>(-1))
            value_count_per_key = value_count_per_key_;
        if (value_count_per_key > value_count_per_key_)
            throw std::runtime_error("value_count_per_key exceeds that in the map.");
        std::string hint;
        hint.append("Fail to serialize ArrayHashMap to \"");
        hint.append(path);
        hint.append("\"; ");
        MapFileHeader header;
        header.fill_basic_fields(optimized_mode);
        header.key_type = static_cast<uint64_t>(DataTypeToCode<TKey>::value);
        header.value_type = static_cast<uint64_t>(DataTypeToCode<TValue>::value);
        header.key_count = key_count_;
        header.bucket_count = bucket_count_;
        header.value_count = value_count_per_key * key_count_;
        header.value_count_per_key = value_count_per_key;
        write(static_cast<const void *>(&header), sizeof(header));
        if (optimized_mode) {
            std::vector<uint8_t> buffer(perfect_hash_get_entry_size(header.value_count_per_key));
            PerfectHashIndexBuilder<TKey> hash_index;
            hash_index.set_keys_array(keys_);
            hash_index.set_key_count(key_count_);
            hash_index.build(hint);
            const auto &entries = hash_index.get_bucket_entries();
            perfect_hash_write_index_entry(write, buffer, hash_index.get_bucket_count(),
                                           hash_index.get_outer_a(), hash_index.get_outer_b(), 1);
            for (uint64_t j = 0; j < hash_index.get_bucket_count(); j++) {
                const auto &e = entries.at(j);
                const std::vector<TKey> &keys = e.keys_;
                const uint64_t mj = keys.size();
                if (mj == 0)
                    perfect_hash_write_zero_entry(write, buffer);
                else if (mj == 1)
                    perfect_hash_write_data_entry(write, buffer, keys.at(0),
                                                  header.value_count_per_key);
                else
                    perfect_hash_write_index_entry(write, buffer, mj, e.a_, e.b_, e.ref_);
            }
            for (uint64_t j = 0; j < hash_index.get_bucket_count(); j++) {
                const auto &e = entries.at(j);
                const std::vector<TKey> &keys = e.keys_;
                const uint64_t mj = keys.size();
                if (mj <= 1)
                    continue;
                const std::vector<bool> &used = e.used_;
                for (uint64_t k = 0; k < mj; k++)
                    if (used.at(k))
                        perfect_hash_write_data_entry(write, buffer, keys.at(k),
                                                      header.value_count_per_key);
                    else
                        perfect_hash_write_zero_entry(write, buffer);
            }
        } else {
            write(static_cast<const void *>(keys_), key_count_ * sizeof(TKey));
            if (value_count_per_key == value_count_per_key_)
                write(static_cast<const void *>(values_), value_count_ * sizeof(TValue));
            else {
                for (uint64_t i = 0; i < key_count_; i++) {
                    const TValue *values = &values_[i * value_count_per_key_];
                    write(static_cast<const void *>(values), value_count_per_key * sizeof(TValue));
                }
            }
            write(static_cast<const void *>(next_), key_count_ * sizeof(uint32_t));
            write(static_cast<const void *>(first_), bucket_count_ * sizeof(uint32_t));
        }
        return key_count_;
    }

    template <typename Func>
    int deserialize_with_header(const std::string &path, Func read, MapFileHeader &header) {
        std::string hint;
        hint.append("Fail to deserialize ArrayHashMap from \"");
        hint.append(path);
        hint.append("\"; ");
        header.validate(header.is_optimized_mode, hint);

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
        value_count_per_key_ = value_count_per_key;
        clear();
        reserve(header.bucket_count);
        if (header.is_optimized_mode == 1) {
            std::vector<uint8_t> buffer(perfect_hash_get_entry_size(value_count_per_key_));
            read(buffer.data(), buffer.size(), hint, "head entry");
            for (uint64_t k = 0; k < header.key_count; k++) {
                for (;;) {
                    read(buffer.data(), buffer.size(), hint, "index, data or zero entry");
                    const uint64_t *const ptr = reinterpret_cast<uint64_t *>(buffer.data());
                    const uint64_t tag = ptr[0];
                    if (tag != 1)
                        continue;
                    const TKey key = static_cast<TKey>(ptr[1]);
                    const TValue *const values = reinterpret_cast<const TValue *>(&ptr[2]);
                    if (contains(key))
                        spdlog::info("duplicate: {}", key);
                    else
                        put(key, values);
                    break;
                }
            }
        } else if (header.bucket_count <= 2 || HashtableHelpers::get_prime_bucket_count(
                                                   header.bucket_count) == header.bucket_count) {
            std::vector<TKey> keys(header.key_count);
            std::vector<TValue> values(value_count);
            read(static_cast<void *>(keys.data()), header.key_count * sizeof(TKey), hint,
                 "keys array");
            read(static_cast<void *>(values.data()), value_count * sizeof(TValue), hint,
                 "values array");
            for (size_t k = 0; k < keys.size(); k++) {
                const TKey key = keys.at(k);
                if (contains(key))
                    spdlog::info("duplicate: {}", key);
                else
                    put(key, values.data() + k * value_count_per_key);
            }
        } else {
            read(static_cast<void *>(keys_), header.key_count * sizeof(TKey), hint, "keys array");
            read(static_cast<void *>(values_), value_count * sizeof(TValue), hint, "values array");
            read(static_cast<void *>(next_), header.key_count * sizeof(uint32_t), hint,
                 "next array");
            read(static_cast<void *>(first_), header.bucket_count * sizeof(uint32_t), hint,
                 "first array");
            key_count_ = header.key_count;
            bucket_count_ = header.bucket_count;
            value_count_ = value_count;
        }
        return key_count_;
    }

    template <typename Func> int deserialize(const std::string &path, Func read) {
        std::string hint;
        hint.append("Fail to deserialize ArrayHashMap from \"");
        hint.append(path);
        hint.append("\"; ");
        MapFileHeader header;
        read(static_cast<void *>(&header), sizeof(header), hint, "map file header");
        return deserialize_with_header(path, std::move(read), header);
    }

    void serialize_to(const std::string &path,
                      uint64_t value_count_per_key = static_cast<uint64_t>(-1),
                      bool optimized_mode = false) {
        FILE *fout = fopen(path.c_str(), "wb");
        if (fout == NULL) {
            std::ostringstream serr;
            serr << "can not open file \"" << path << "\" for map serializing; ";
            serr << strerror(errno);
            throw std::runtime_error(serr.str());
        }
        std::unique_ptr<FILE, decltype(&fclose)> fout_guard(fout, &fclose);
        serialize(
            path, [fout](const void *ptr, size_t size) { fwrite(ptr, 1, size, fout); },
            value_count_per_key, optimized_mode);
    }

    void deserialize_from(const std::string &path) {
        std::string hint;
        hint.append("Fail to deserialize ArrayHashMap from \"");
        hint.append(path);
        hint.append("\"; ");
        FILE *fin = fopen(path.c_str(), "rb");
        if (fin == NULL) {
            std::ostringstream serr;
            serr << hint;
            serr << "can not open file. ";
            serr << strerror(errno);
            throw std::runtime_error(serr.str());
        }
        uint64_t offset = 0;
        std::unique_ptr<FILE, decltype(&fclose)> fin_guard(fin, &fclose);
        deserialize(path, [fin, &offset](void *ptr, size_t size, const std::string &hint,
                                         const std::string &what) {
            const size_t nread = fread(ptr, 1, size, fin);
            if (nread != size) {
                std::ostringstream serr;
                serr << hint;
                serr << "incomplete " << what << ", ";
                serr << size << " bytes expected, ";
                serr << "but only " << nread << " are read successfully. ";
                serr << "offset = " << offset << " (0x" << std::hex << offset << ")";
                throw std::runtime_error(serr.str());
            }
            offset += nread;
        });
    }

  private:
    static uint64_t perfect_hash_get_entry_size(uint64_t value_count_per_key) {
        const uint64_t size1 = sizeof(uint64_t) * 4;
        const uint64_t size2 = sizeof(uint64_t) * 2 + sizeof(TValue) * value_count_per_key;
        const uint64_t size = size1 > size2 ? size1 : size2;
        const uint64_t mask = sizeof(uint64_t) - 1;
        const uint64_t buffer_size = (size + mask) & ~mask;
        return buffer_size;
    }

    template <typename Func>
    static void perfect_hash_write_zero_entry(Func write, std::vector<uint8_t> &buffer) {
        const uint64_t buffer_size = buffer.size();
        memset(buffer.data(), 0, buffer_size);
        write(buffer.data(), buffer_size);
    }

    template <typename Func>
    static void perfect_hash_write_index_entry(Func write, std::vector<uint8_t> &buffer, uint64_t m,
                                               uint64_t a, uint64_t b, uint64_t r) {
        const uint64_t buffer_size = buffer.size();
        memset(buffer.data(), 0, buffer_size);
        uint64_t *const ptr = reinterpret_cast<uint64_t *>(buffer.data());
        ptr[0] = m;
        ptr[1] = a;
        ptr[2] = b;
        ptr[3] = r;
        write(buffer.data(), buffer_size);
    }

    template <typename Func>
    void perfect_hash_write_data_entry(Func write, std::vector<uint8_t> &buffer, TKey key,
                                       uint64_t value_count_per_key) {
        const uint64_t buffer_size = buffer.size();
        memset(buffer.data(), 0, buffer_size);
        uint64_t *const ptr = reinterpret_cast<uint64_t *>(buffer.data());
        ptr[0] = static_cast<uint64_t>(1);
        ptr[1] = static_cast<uint64_t>(key);
        TValue *const values = reinterpret_cast<TValue *>(&ptr[2]);
        const TValue *const data = get(key);
        memcpy(values, data, sizeof(TValue) * value_count_per_key);
        write(buffer.data(), buffer_size);
    }

    void build_hash_index() {
        memset(first_, -1, bucket_count_ * sizeof(uint32_t));
        for (uint64_t i = 0; i < key_count_; i++) {
            const TKey key = keys_[i];
            const uint64_t bucket = get_bucket(key);
            next_[i] = first_[bucket];
            first_[bucket] = static_cast<uint32_t>(i);
        }
    }

    void ensure_capacity() {
        uint64_t min_capacity = key_count_ * 2;
        if (min_capacity == 0)
            min_capacity = 1000;
        uint64_t size = HashtableHelpers::get_power_bucket_count(min_capacity);
        uint64_t capacity = size;
        if (capacity < min_capacity)
            capacity = min_capacity;
        reserve(capacity);
    }

    static constexpr uint64_t fast_modulo(uint64_t a) {
        constexpr int q = 31;
        constexpr uint64_t prime = (UINT64_C(1) << q) - 1;
        uint64_t r = (a & prime) + (a >> q);
        if (r >= prime)
            r -= prime;
        return r;
    }

    MemoryBuffer keys_buffer_;
    MemoryBuffer values_buffer_;
    MemoryBuffer next_buffer_;
    MemoryBuffer first_buffer_;
    uint64_t key_count_ = 0;
    uint64_t bucket_count_ = 0;
    uint64_t value_count_ = 0;
    uint64_t value_count_per_key_ = static_cast<uint64_t>(-1);
    TKey *keys_ = nullptr;
    TValue *values_ = nullptr;
    uint32_t *next_ = nullptr;
    uint32_t *first_ = nullptr;
};

} // namespace metaspore