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
#include <iostream>
#include <memory>
#include <mmintrin.h>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <tuple>
#include <utility>
#include <xmmintrin.h>

namespace metaspore {

template <typename TKey, typename TValue> class PerfectArrayHashMap {
  public:
    struct Entry {
      public:
        friend class PerfectArrayHashMap;

      private:
        uint64_t tag;
        union {
            struct {
                uint64_t a;
                uint64_t b;
                uint64_t ref;
            } index;
            struct {
                uint64_t key;
            } data;
        };

        TKey get_key() const { return static_cast<TKey>(data.key); }

        const TValue *get_values() const {
            const char *p = reinterpret_cast<const char *>(this);
            const char *ptr = p + sizeof(uint64_t) * 2;
            return reinterpret_cast<const TValue *>(ptr);
        }
    };

  public:
    explicit PerfectArrayHashMap(std::shared_ptr<void> blob) {
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
        value_count_ = value_count;
        value_count_per_key_ = value_count_per_key;
        const char *address = static_cast<const char *>(ptr);
        address += sizeof(MapFileHeader);
        entries_ = reinterpret_cast<const Entry *>(address);
        const Entry *const e = get_entry_ptr(0);
        bucket_count_ = e->tag;
        outer_a_ = e->index.a;
        outer_b_ = e->index.b;
        blob_ = std::move(blob);
    }

    uint64_t get_key_count() const { return key_count_; }
    uint64_t get_bucket_count() const { return bucket_count_; }
    uint64_t get_value_count() const { return value_count_; }
    uint64_t get_value_count_per_key() const { return value_count_per_key_; }

    uint64_t size() const { return key_count_; }

    bool empty() const { return key_count_ == 0; }

    bool contains(TKey key) const {
        const uint64_t j =
            1 + HashtableHelpers::universal_hash(key, outer_a_, outer_b_, bucket_count_);
        const Entry *const entry = get_entry_ptr(j);
        const uint64_t tag = entry->tag;
        if (tag == 0)
            return false;
        if (tag == 1)
            return entry->get_key() == key;
        const uint64_t mj = tag;
        const uint64_t a = entry->index.a;
        const uint64_t b = entry->index.b;
        const uint64_t ref = entry->index.ref;
        const uint64_t k = ref + HashtableHelpers::universal_hash(key, a, b, mj);
        const Entry *const e = get_entry_ptr(k);
        return e->tag == 1 && e->get_key() == key;
    }

    int64_t find(TKey key) const {
        const uint64_t j =
            1 + HashtableHelpers::universal_hash(key, outer_a_, outer_b_, bucket_count_);
        const Entry *const entry = get_entry_ptr(j);
        const uint64_t tag = entry->tag;
        if (tag == 0)
            return -1;
        if (tag == 1)
            return entry->get_key() == key ? static_cast<int64_t>(j) : -1;
        const uint64_t mj = tag;
        const uint64_t a = entry->index.a;
        const uint64_t b = entry->index.b;
        const uint64_t ref = entry->index.ref;
        const uint64_t k = ref + HashtableHelpers::universal_hash(key, a, b, mj);
        const Entry *const e = get_entry_ptr(k);
        return e->tag == 1 && e->get_key() == key ? static_cast<int64_t>(k) : -1;
    }

    /**
     *  Get one value by key, with no prefetch by default.
     */
    template <bool prefetch = false> const TValue *get(TKey key) const {
        const Entry *const entry = get_entry_ptr_j_with_prefetch<prefetch>(key);
        const auto entry_result_tuple =
            get_entry_ptr_k_or_value_with_prefetch<prefetch>(key, entry);
        return get_value_ptr_with_prefetch(key, entry_result_tuple);
    }

    template <bool prefetch = true>
    inline const Entry *get_entry_ptr_j_with_prefetch(TKey key) const {
        const uint64_t j =
            1 + HashtableHelpers::universal_hash(key, outer_a_, outer_b_, bucket_count_);
        const auto *ptr = get_entry_ptr(j);
        if constexpr (prefetch)
            _mm_prefetch(ptr, _MM_HINT_NTA);
        return ptr;
    }

    template <bool prefetch = true>
    inline std::tuple<const Entry *, const TValue *>
    get_entry_ptr_k_or_value_with_prefetch(TKey key, const Entry *entry) const {
        const uint64_t tag = entry->tag;
        if (tag == 0)
            return {nullptr, nullptr};
        if (tag == 1)
            return {nullptr, entry->get_key() == key ? entry->get_values() : nullptr};
        const uint64_t mj = tag;
        const uint64_t a = entry->index.a;
        const uint64_t b = entry->index.b;
        const uint64_t ref = entry->index.ref;
        const uint64_t k = ref + HashtableHelpers::universal_hash(key, a, b, mj);
        const Entry *const e = get_entry_ptr(k);
        if constexpr (prefetch)
            _mm_prefetch(e, _MM_HINT_NTA);
        return {e, nullptr};
    }

    inline static const TValue *
    get_value_ptr_with_prefetch(TKey key,
                                const std::tuple<const Entry *, const TValue *> &entry_k_or_value) {
        auto [e, value] = entry_k_or_value;
        if (e) {
            return e->tag == 1 && e->get_key() == key ? e->get_values() : nullptr;
        } else {
            return value;
        }
    }

    template <typename Iterator, uint64_t N>
    void get_n(Iterator begin, Iterator end, std::array<const TValue *, N> &result) const {
        std::array<const Entry *, N> entry_array;
        for (size_t i = 0; i < N; ++i) {
            TKey key = *(begin + i);
            entry_array[i] = get_entry_ptr_j_with_prefetch(key);
        }
        std::array<std::tuple<const Entry *, const float *>, N> entry_k_or_result_array;
        for (size_t i = 0; i < N; ++i) {
            TKey key = *(begin + i);
            const Entry *const entry = entry_array[i];
            entry_k_or_result_array[i] = get_entry_ptr_k_or_value_with_prefetch(key, entry);
        }
        for (size_t i = 0; i < N; ++i) {
            TKey key = *(begin + i);
            result[i] = get_value_ptr_with_prefetch(key, entry_k_or_result_array[i]);
        }
    }

    static int64_t choose_batch_size(size_t total) {
        if (total >= 128) {
            return 128;
        }
        if (total >= 64) {
            return 64;
        }
        if (total >= 32) {
            return 32;
        }
        if (total >= 16) {
            return 16;
        }
        if (total >= 8) {
            return 8;
        }
        return total;
    }

#define CASE(n)                                                                                    \
    case n: {                                                                                      \
        struct N {                                                                                 \
            static constexpr auto value() { return n##UL; }                                        \
        };                                                                                         \
        while (left >= batch_size) {                                                               \
            fn(begin, begin + batch_size, N{});                                                    \
            begin += batch_size;                                                                   \
            left -= batch_size;                                                                    \
        }                                                                                          \
        break;                                                                                     \
    }

    /**
     *   A helper function to invoke @fn by batch with proper batch size while
     *   iterating between [@begin, @end).
     *
     *   The batch size will be automatically decided (by choose_batch_size).
     *
     *   Since explicit templated lambda is only available in C++20, we use an
     *   auto parameter for @fn to deduce batch size at compile time so as to make
     *   this code C++17 compatible.
     *
     *   @fn will receive 3 parameters:
     *       1. Iterator begin, indicate the beginning of the *current* batch;
     *       2. Iterator end, indicate the end of the *current* batch (not included);
     *       3. A struct with a static member method value to get a constexpr size of this batch.
     *
     *   It is suggested to write a templated lambda fn like the following:
     *   auto fn = [](Iterator begin, Iterator end, auto n) {
     *       constexpr size_t N = decltype(n)::value();
     *       // ... your logics here. N can be used as template parameter.
     *   };
     */
    template <typename Iterator, typename Func>
    static void invoke_by_batch(Iterator begin, const Iterator end, Func &&fn) {
        while (begin < end) {
            int64_t left = end - begin;
            const int64_t batch_size = choose_batch_size(left);

            switch (batch_size) {
                CASE(64)
                CASE(32)
                CASE(16)
                CASE(8)
                CASE(7)
                CASE(6)
                CASE(5)
                CASE(4)
                CASE(3)
                CASE(2)
                CASE(1)
            default:
                __builtin_unreachable();
            } // switch
        }     // while
    }

#undef CASE

    template <typename Iterator, typename Func>
    void get_batch(Iterator begin, const Iterator end, Func &&fn) {
        auto invoke_fn = [fn = std::move(fn), this](Iterator begin, Iterator end, auto n) mutable {
            constexpr size_t N = decltype(n)::value();
            std::array<const TValue *, N> result;
            get_n(begin, begin + N, result);
            for (const TValue *ptr : result) {
                fn(ptr);
            }
        };
        invoke_by_batch(begin, end, std::move(invoke_fn));
    }

    const TValue *get(TKey key, uint64_t &value_count) const {
        const TValue *const ptr = get(key);
        value_count = ptr ? value_count_per_key_ : 0;
        return ptr;
    }

    void dump(std::ostream &out = std::cerr, uint64_t count_limit = uint64_t(-1)) const {
        uint64_t i = 0;
        for (uint64_t k = 0; k < key_count_; k++) {
            if (k >= count_limit)
                break;
            for (;;) {
                const Entry *e = get_entry_ptr(++i);
                if (e->tag != 1)
                    continue;
                const TKey key = e->get_key();
                const TValue *const values = e->get_values();
                out << key << ": [";
                for (uint64_t j = 0; j < value_count_per_key_; j++)
                    out << (j ? ", " : "") << as_number(values[j]);
                out << "]\n";
                break;
            }
        }
    }

    template <typename Func> void each(Func action) {
        uint64_t i = 0;
        for (uint64_t k = 0; k < key_count_; k++) {
            for (;;) {
                const Entry *e = get_entry_ptr(++i);
                if (e->tag != 1)
                    continue;
                const TKey key = e->get_key();
                const TValue *const values = e->get_values();
                action(i, key, values, value_count_per_key_);
                break;
            }
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

    class iterator {
      public:
        iterator(const PerfectArrayHashMap<TKey, TValue> *map, uint64_t index)
            : map_(map), index_(index) {
            if (index_ < map_->key_count_)
                advance_entry_pointer();
        }

        iterator &operator++() {
            if (index_ < map_->key_count_) {
                index_++;
                if (index_ < map_->key_count_)
                    advance_entry_pointer();
            }
            return *this;
        }

        TKey operator*() const {
            if (index_ < map_->key_count_)
                return key_;
            else
                return TKey(-1);
        }

        bool operator==(const iterator &rhs) const {
            return index_ == rhs.index_ && map_ == rhs.map_;
        }

        bool operator!=(const iterator &rhs) const { return !(*this == rhs); }

      private:
        void advance_entry_pointer() {
            for (;;) {
                const Entry *const entry = map_->get_entry_ptr(++i_);
                if (entry->tag != 1)
                    continue;
                key_ = entry->get_key();
                break;
            }
        }

        const PerfectArrayHashMap<TKey, TValue> *map_;
        uint64_t index_;
        uint64_t i_ = 0;
        TKey key_ = TKey(-1);
    };

    iterator begin() const { return iterator(this, 0); }

    iterator end() const { return iterator(this, key_count_); }

    uint64_t get_entry_size() const {
        const uint64_t size1 = sizeof(uint64_t) * 4;
        const uint64_t size2 = sizeof(uint64_t) * 2 + sizeof(TValue) * value_count_per_key_;
        const uint64_t size = size1 > size2 ? size1 : size2;
        const uint64_t mask = sizeof(uint64_t) - 1;
        const uint64_t buffer_size = (size + mask) & ~mask;
        return buffer_size;
    }

    const Entry *get_entry_ptr(uint64_t i) const {
        const char *p = reinterpret_cast<const char *>(entries_);
        const char *ptr = p + get_entry_size() * i;
        return reinterpret_cast<const Entry *>(ptr);
    }

    std::shared_ptr<void> blob_;
    uint64_t key_count_;
    uint64_t bucket_count_;
    uint64_t outer_a_;
    uint64_t outer_b_;
    uint64_t value_count_;
    uint64_t value_count_per_key_;
    const Entry *entries_;
};

} // namespace metaspore