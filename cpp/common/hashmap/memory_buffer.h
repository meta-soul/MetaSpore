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

#include <algorithm>
#include <new>
#include <stdint.h>
#include <stdlib.h>

namespace metaspore {

class MemoryBuffer {
  public:
    MemoryBuffer() {
        ptr_ = nullptr;
        size_ = 0;
    }

    explicit MemoryBuffer(uint64_t size) {
        if (size == 0) {
            ptr_ = nullptr;
            size_ = 0;
        } else {
            ptr_ = malloc(size);
            if (!ptr_)
                throw std::bad_alloc();
            size_ = size;
        }
    }

    MemoryBuffer(MemoryBuffer &&rhs) {
        ptr_ = rhs.ptr_;
        size_ = rhs.size_;
        rhs.ptr_ = nullptr;
        rhs.size_ = 0;
    }

    ~MemoryBuffer() {
        if (ptr_) {
            free(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    void swap(MemoryBuffer &other) {
        std::swap(ptr_, other.ptr_);
        std::swap(size_, other.size_);
    }

    void *get_pointer() const { return ptr_; }

    uint64_t get_size() const { return size_; }

    void allocate(uint64_t size) {
        MemoryBuffer buf(size);
        swap(buf);
    }

    void deallocate() {
        MemoryBuffer buf;
        swap(buf);
    }

    void reallocate(uint64_t size) {
        if (size == 0)
            deallocate();
        else {
            void *new_ptr = realloc(ptr_, size);
            if (!new_ptr)
                throw std::bad_alloc();
            ptr_ = new_ptr;
            size_ = size;
        }
    }

    void reallocate_fill(uint64_t size, int value) {
        const uint64_t old_size = size_;
        reallocate(size);
        if (old_size < size)
            memset(static_cast<char *>(ptr_) + old_size, value, size - old_size);
    }

    void reallocate_clear(uint64_t size) { reallocate_fill(size, 0); }

    void fill(int value) {
        if (ptr_)
            memset(ptr_, value, size_);
    }

    void clear() { fill(0); }

  private:
    void *ptr_;
    uint64_t size_;
};

} // namespace metaspore