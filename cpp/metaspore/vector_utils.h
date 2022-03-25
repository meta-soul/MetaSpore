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

#include <stdint.h>
#include <vector>

//
// ``vector_utils.h`` defines utility functions for ``std::vector``.
//

namespace metaspore {

template <typename T, typename InputIterator>
inline void VectorAppend(std::vector<T> &v, InputIterator first, InputIterator last) {
    v.insert(v.end(), first, last);
}

template <typename T, typename RandomAccessIterator>
inline void VectorAppend(std::vector<T> &v, RandomAccessIterator first, size_t count) {
    VectorAppend(v, first, first + count);
}

template <typename T> struct VectorBase {
    T *start;
    T *finish;
    T *end_of_storage;

    VectorBase() : start(), finish(), end_of_storage() {}
};

template <typename T>
inline void VectorDetachBuffer(std::vector<T> &v, T *&data, size_t &size, size_t &capacity) {
    VectorBase<T> base;
    std::vector<T> &fake = reinterpret_cast<std::vector<T> &>(base);
    fake.swap(v);
    data = base.start;
    size = base.finish - base.start;
    capacity = base.end_of_storage - base.start;
}

template <typename T>
inline void VectorAttachBuffer(std::vector<T> &v, T *data, size_t size, size_t capacity) {
    VectorBase<T> base;
    base.start = data;
    base.finish = data + size;
    base.end_of_storage = data + capacity;
    std::vector<T> &fake = reinterpret_cast<std::vector<T> &>(base);
    std::vector<T> t;
    fake.swap(t);
    t.swap(v);
}

} // namespace metaspore
