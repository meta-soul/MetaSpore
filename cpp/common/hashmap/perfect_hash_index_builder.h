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

#include <chrono>
#include <common/hashmap/hashtable_helpers.h>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace metaspore {

template <typename TKey> class PerfectHashIndexBuilder {
  public:
    struct BucketEntry {
        uint64_t a_;
        uint64_t b_;
        std::vector<TKey> keys_;
        std::vector<bool> used_;
        uint64_t ref_;
    };

    const TKey *get_keys_array() const { return keys_array_; }
    uint64_t get_key_count() const { return key_count_; }
    uint64_t get_bucket_count() const { return bucket_count_; }
    uint64_t get_outer_a() const { return outer_a_; }
    uint64_t get_outer_b() const { return outer_b_; }
    int get_outer_tries() const { return outer_tries_; }
    int get_inner_tries() const { return inner_tries_; }
    const std::vector<BucketEntry> &get_bucket_entries() const { return entries_; }

    void set_keys_array(const TKey *keys_array) { keys_array_ = keys_array; }
    void set_outer_tries(int outer_tries) { outer_tries_ = outer_tries; }
    void set_inner_tries(int inner_tries) { inner_tries_ = inner_tries; }

    void set_key_count(uint64_t key_count) {
        const uint64_t limit = std::numeric_limits<int64_t>::max();
        if (key_count > limit) {
            std::ostringstream serr;
            serr << "can not store " << key_count << " keys, ";
            serr << "at most " << limit << " are allowed.";
            throw std::runtime_error(serr.str());
        }
        key_count_ = key_count;
    }

    void build(const std::string &hint) {
        if (!try_build()) {
            std::ostringstream serr;
            serr << hint;
            serr << "unable to build perfect hash index. ";
            serr << "key_count = " << key_count_ << ", ";
            serr << "bucket_count = " << bucket_count_ << ", ";
            serr << "outer_tries = " << outer_tries_ << ", ";
            serr << "inner_tries = " << inner_tries_ << ".";
            throw std::runtime_error(serr.str());
        }
    }

    bool try_build() {
        const size_t num = (3 * key_count_ + 1) / 2;
        bucket_count_ = HashtableHelpers::get_power_bucket_count(num);
        if (bucket_count_ == 0)
            bucket_count_ = 1;
        const uint64_t upper_bound = std::numeric_limits<uint64_t>::max();
        const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);
        std::uniform_int_distribution<uint64_t> distribution(0, upper_bound);
        for (int round = 1; round <= outer_tries_; round++) {
            buckets_.clear();
            entries_.clear();
            buckets_.resize(bucket_count_);
            entries_.resize(bucket_count_);
            const uint64_t a = distribution(generator);
            const uint64_t b = distribution(generator);
            for (uint64_t k = 0; k < key_count_; k++) {
                const TKey key = keys_array_[k];
                const uint64_t h = HashtableHelpers::universal_hash(key, a, b, bucket_count_);
                buckets_.at(h).push_back(key);
            }
            uint64_t refs = 1 + bucket_count_;
            uint64_t j;
            for (j = 0; j < bucket_count_; j++) {
                std::vector<TKey> &keys = buckets_.at(j);
                const uint64_t nj = keys.size();
                if (nj <= 1) {
                    BucketEntry entry;
                    entry.a_ = entry.b_ = entry.ref_ = 0;
                    if (nj == 1)
                        entry.keys_.swap(keys);
                    entries_.at(j) = std::move(entry);
                    continue;
                }
                const uint64_t mj = HashtableHelpers::get_power_bucket_count(nj * nj);
                const uint64_t inner_tries = mj > uint64_t(inner_tries_) ? mj : inner_tries_;
                BucketEntry e;
                uint64_t tries;
                for (tries = 1; tries <= inner_tries; tries++) {
                    e.a_ = distribution(generator);
                    e.b_ = distribution(generator);
                    e.keys_.clear();
                    e.keys_.resize(mj);
                    e.used_.clear();
                    e.used_.resize(mj);
                    uint64_t k;
                    for (k = 0; k < nj; k++) {
                        const TKey key = keys.at(k);
                        const uint64_t h = HashtableHelpers::universal_hash(key, e.a_, e.b_, mj);
                        if (e.used_.at(h))
                            break;
                        e.keys_.at(h) = key;
                        e.used_.at(h) = true;
                    }
                    if (k >= nj) {
                        e.ref_ = refs;
                        entries_.at(j) = std::move(e);
                        keys.clear();
                        refs += mj;
                        break;
                    }
                }
                if (tries > inner_tries_)
                    break;
            }
            if (j >= bucket_count_) {
                outer_a_ = a;
                outer_b_ = b;
                return true;
            }
        }
        return false;
    }

  private:
    const TKey *keys_array_ = nullptr;
    uint64_t key_count_ = 0;
    uint64_t bucket_count_ = 0;
    uint64_t outer_a_ = 0;
    uint64_t outer_b_ = 0;
    int outer_tries_ = 100;
    int inner_tries_ = 1000;
    std::vector<std::vector<TKey>> buckets_;
    std::vector<BucketEntry> entries_;
};

} // namespace metaspore