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

#include <iterator>
#include <tuple>

namespace metaspore {

template <typename T, typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr static auto enumerate(T &&iterable) {
    struct iterator {
        size_t i;
        TIter iter;
        bool operator!=(const iterator &other) const { return iter != other.iter; }
        void operator++() {
            ++i;
            ++iter;
        }
        auto operator*() const { return std::tie(i, *iter); }
    };
    struct iterable_wrapper {
        T &&iterable;
        auto begin() { return iterator{0, std::begin(iterable)}; }
        auto end() { return iterator{0, std::end(iterable)}; }
    };
    return iterable_wrapper{std::forward<T>(iterable)};
}

template <typename... T> struct iterator_helper {
    using IterableTuple = std::tuple<T...>;
    using IteratorTuple = std::tuple<decltype(std::begin(std::declval<T>()))...>;

    template <size_t... N> static void incr(IteratorTuple &iters, std::index_sequence<N...>) {
        (void)std::initializer_list<int>{((void)++std::get<N>(iters), 0)...};
    }

    template <size_t... N> static auto deref(IteratorTuple &iters, std::index_sequence<N...>) {
        return std::forward_as_tuple(*std::get<N>(iters)...);
    }

    template <size_t... N> static auto begin(IterableTuple &iterables, std::index_sequence<N...>) {
        return std::make_tuple(std::begin(std::get<N>(iterables))...);
    }

    template <size_t... N> static auto end(IterableTuple &iterables, std::index_sequence<N...>) {
        return std::make_tuple(std::end(std::get<N>(iterables))...);
    }
};

template <typename... T> constexpr static auto zip(std::tuple<T...> iterable) {
    using IterableTuple = std::tuple<T...>;
    using IteratorTuple = std::tuple<decltype(std::begin(std::declval<T>()))...>;
    constexpr size_t n = std::tuple_size<IterableTuple>::value;
    using idx_t = std::make_index_sequence<n>;

    struct iterator {
        IteratorTuple iters;
        bool operator!=(const iterator &other) const { return iters != other.iters; }
        void operator++() { iterator_helper<T...>::template incr(iters, idx_t{}); }
        auto operator*() { return iterator_helper<T...>::template deref(iters, idx_t{}); }
    };

    struct iterable_wrapper {
        IterableTuple iterable;
        auto begin() { return iterator{iterator_helper<T...>::template begin(iterable, idx_t{})}; }
        auto end() { return iterator{iterator_helper<T...>::template end(iterable, idx_t{})}; }
    };

    return iterable_wrapper{iterable};
}

} // namespace metaspore