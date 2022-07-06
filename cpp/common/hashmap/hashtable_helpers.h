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

#include <iosfwd>
#include <stdint.h>

namespace metaspore {

class HashtableHelpers {
  public:
    static uint64_t get_prime_bucket_count(uint64_t min_size);

    static constexpr uint64_t get_power_bucket_count(uint64_t v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v++;
        return v;
    }

    template <typename TKey>
    static constexpr uint64_t universal_hash(TKey key, uint64_t a, uint64_t b, uint64_t m) {
        const uint64_t k = static_cast<uint64_t>(key);
        const __uint128_t c = a * static_cast<__uint128_t>(k) + b;
        const __uint128_t h = fast_modulo(c, large_prime) & (m - 1);
        return static_cast<uint64_t>(h);
    }

    // A Mersenne prime 2^89 - 1.
    // See: High Speed Hashing for Integers and Strings (https://arxiv.org/pdf/1504.06804.pdf)
    // Page 5, 2.2.1 Implementation for 64-bit keys
    // q = 89
    static constexpr __uint128_t large_prime = __uint128_t(0x1ffffff) << 64 | 0xffffffffffffffff;

    static size_t get_peak_rss();
    static size_t get_current_rss();
    static void show_memory_usage();

    static constexpr uint64_t FastModulo(uint64_t a) {
        constexpr int q = 31;
        constexpr uint64_t prime = (UINT64_C(1) << q) - 1;
        uint64_t r = (a & prime) + (a >> q);
        if (r >= prime)
            r -= prime;
        return r;
    }

  private:
    static constexpr __uint128_t fast_modulo(__uint128_t a, __uint128_t b) {
        constexpr int q = 89;
        uint64_t r = (a & b) + (a >> q);
        if (r >= b)
            r -= b;
        return r;
    }
};

std::ostream &operator<<(std::ostream &sout, __uint128_t value);

} // namespace metaspore