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

#include <common/hashmap/memory_mapped_array_hash_map.h>
#include <common/hashmap/perfect_array_hash_map.h>
#include <emmintrin.h>
#include <memory>
#include <vector>

namespace metaspore {

template <typename TKey, typename TValue, typename MapContainer> struct MapContainerAdapter;

template <typename TKey, typename TValue, template <typename, typename> typename MapType>
struct MapContainerAdapter<TKey, TValue, std::vector<MapType<TKey, TValue>>> {
    using ContainerType = std::vector<MapType<TKey, TValue>>;
    static const MapType<TKey, TValue> *get_map(const ContainerType &maps, size_t i) {
        return &maps[i];
    }
};

template <typename TKey, typename TValue, template <typename, typename> typename MapType>
struct MapContainerAdapter<TKey, TValue, std::vector<std::shared_ptr<MapType<TKey, TValue>>>> {
    using ContainerType = std::vector<std::shared_ptr<MapType<TKey, TValue>>>;
    static const MapType<TKey, TValue> *get_map(const ContainerType &maps, size_t i) {
        return maps[i].get();
    }
};

template <typename TKey, typename TValue, template <typename, typename> typename MapType>
struct MapContainerAdapter<TKey, TValue, std::vector<std::unique_ptr<MapType<TKey, TValue>>>> {
    using ContainerType = std::vector<std::unique_ptr<MapType<TKey, TValue>>>;
    static const MapType<TKey, TValue> *get_map(const ContainerType &maps, size_t i) {
        return maps[i].get();
    }
};

template <typename TKey, typename TValue, typename MapContainer, typename Iterator, typename Func>
class MultiPerfectArrayHashMapSearcher {
  public:
    using MapContainerAdapterT = MapContainerAdapter<TKey, TValue, MapContainer>;
    using MapType = PerfectArrayHashMap<TKey, TValue>;

    static void search(Iterator begin, Iterator end, const MapContainer &maps, Func &&func) {
        MapType::invoke_by_batch(
            begin, end, [func = std::move(func), &maps](auto begin, auto end, auto n) mutable {
                constexpr size_t N = decltype(n)::value();
                const size_t slicecount = maps.size();

                std::array<const typename MapType::Entry *, N> entry_j_array;
                std::array<std::pair<TKey, const MapType *>, N> key_map_array;
                for (size_t i = 0; i < N; ++i) {
                    const TKey key = *(begin + i);
                    const uint64_t slice_no = key % slicecount;
                    const MapType *map = MapContainerAdapterT::get_map(maps, slice_no);
                    key_map_array[i] = {key, map};
                    entry_j_array[i] = map->get_entry_ptr_j_with_prefetch(key);
                }
                _mm_pause();

                std::array<std::tuple<const typename MapType::Entry *, const TValue *>, N>
                    entry_k_or_result_array;
                for (size_t i = 0; i < N; ++i) {
                    const auto [key, map] = key_map_array[i];
                    entry_k_or_result_array[i] =
                        map->get_entry_ptr_k_or_value_with_prefetch(key, entry_j_array[i]);
                }
                _mm_pause();

                std::array<const TValue *, N> result_array;
                for (size_t i = 0; i < N; ++i) {
                    const TKey key = *(begin + i);
                    result_array[i] =
                        MapType::get_value_ptr_with_prefetch(key, entry_k_or_result_array[i]);
                }

                for (size_t i = 0; i < N; ++i) {
                    const auto [key, map] = key_map_array[i];
                    const TValue *v = result_array[i];
                    func(key, v, begin + i);
                }
            });
    }
};

template <typename TKey, typename TValue, typename MapContainer, typename Iterator, typename Func>
class MultiMemoryMappedHashMapSearcher {
  public:
    using MapContainerAdapterT = MapContainerAdapter<TKey, TValue, MapContainer>;
    using MapType = MemoryMappedArrayHashMap<TKey, TValue>;
    static void search(Iterator begin, Iterator end, const MapContainer &maps, Func &&func) {
        size_t count = 0UL;
        for (auto it = begin; it != end; ++it) {
            auto key = *it;
            const size_t slicecount = maps.size();
            const uint64_t slice_no = key % slicecount;
            const MapType *map = MapContainerAdapterT::get_map(maps, slice_no);
            const TValue *p = map->get(key);
            func(key, p, count);
            ++count;
        }
    }
};

} // namespace metaspore