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

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace metaspore {

inline uint64_t BKDRHash(const char *str, size_t len, uint64_t seed) {
    for (size_t i = 0; i < len; i++)
        seed = seed * 131 + str[i];
    return seed;
}

inline uint64_t BKDRHash(std::string_view str) { return BKDRHash(str.data(), str.size(), 0); }

inline uint64_t BKDRHashWithEqualPostfix(const char *str, size_t len, uint64_t seed) {
    seed = BKDRHash(str, len, seed);
    seed = seed * 131 + '=';
    return seed;
}

inline uint64_t BKDRHashWithEqualPostfix(std::string_view str) {
    return BKDRHashWithEqualPostfix(str.data(), str.size(), 0);
}

inline constexpr uint64_t CombineHashCodes(uint64_t h, uint64_t x) {
    return h ^ (x + 0x9e3779b9 + (h << 6) + (h >> 2));
}

inline uint64_t BKDRHashOneField(uint64_t name, uint64_t value) {
    return CombineHashCodes(name, value);
}

inline uint64_t BKDRHashConcatOneField(uint64_t first, uint64_t second) {
    return CombineHashCodes(first, second);
}

inline uint64_t BKDRHashWithSeparatePrefixAndEqualPostfix(const char *str, size_t len,
                                                          uint64_t seed) {
    seed = seed * 131 + '\001';
    seed = BKDRHash(str, len, seed);
    seed = seed * 131 + '=';
    return seed;
}

template <typename HashListAccessor, template <typename> typename Container, typename AppendFunc>
class CartesianHashCombine {
  public:
    static void CombineOneFeature(const Container<HashListAccessor> &lists, AppendFunc &&func,
                                  size_t total_results) {
        if (total_results == 1) {
            // each list contains only one element
            uint64_t h = lists[0][0];
            for (size_t i = 1; i < lists.size(); ++i) {
                h = BKDRHashConcatOneField(h, lists[i][0]);
            }
            func(h);
        } else if (lists.size() == 1) {
            // only one list, just append all hash values in it
            for (auto hash : lists[0]) {
                func(hash);
            }
        } else {
            static thread_local std::vector<size_t> prd_fwd(64);
            static thread_local std::vector<size_t> prd_bwd(64);
            static thread_local std::vector<uint64_t> result(64);
            prd_fwd.clear();
            prd_bwd.clear();
            prd_fwd.resize(lists.size());
            prd_bwd.resize(lists.size());
            result.clear();
            result.resize(total_results);
            prd_fwd.at(0) = 1;
            for (size_t i = 1; i < lists.size(); i++)
                prd_fwd[i] = prd_fwd[i - 1] * lists[i - 1].size();
            prd_bwd[lists.size() - 1] = 1;
            for (size_t i = lists.size() - 1; i > 0; i--)
                prd_bwd[i - 1] = prd_bwd[i] * lists[i].size();

            const HashListAccessor &list = lists[0];
            const size_t loops = prd_fwd[0];
            const size_t each_repeat = prd_bwd[0];
            for (size_t l = 0; l < loops; l++) {
                size_t base = l * list.size() * each_repeat;
                for (auto h : list) {
                    for (size_t r = 0; r < each_repeat; r++)
                        result[base + r] = h;
                    base += each_repeat;
                }
            }

            for (size_t i = 1; i < lists.size(); i++) {
                const HashListAccessor &list = lists[i];
                const size_t loops = prd_fwd[i];
                const size_t each_repeat = prd_bwd[i];
                for (size_t l = 0; l < loops; l++) {
                    size_t base = l * list.size() * each_repeat;
                    for (auto hash : list) {
                        for (size_t r = 0; r < each_repeat; r++) {
                            uint64_t &h = result[base + r];
                            h = BKDRHashConcatOneField(h, hash);
                        }
                        base += each_repeat;
                    }
                }
            }
            for (uint64_t h : result) {
                func(h);
            }
        }
    }
};

} // namespace metaspore