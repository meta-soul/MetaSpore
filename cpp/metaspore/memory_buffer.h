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

//
// ``memory_buffer.h`` defines class ``MemoryBuffer`` which simplifies
// the calling of C memory management functions in ``ArrayHashMap``.
//

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

    void Swap(MemoryBuffer &other) {
        std::swap(ptr_, other.ptr_);
        std::swap(size_, other.size_);
    }

    void *GetPointer() const { return ptr_; }

    uint64_t GetSize() const { return size_; }

    void Deallocate() {
        MemoryBuffer buf;
        Swap(buf);
    }

    void Reallocate(uint64_t size) {
        if (size == 0)
            Deallocate();
        else {
            void *new_ptr = realloc(ptr_, size);
            if (!new_ptr)
                throw std::bad_alloc();
            ptr_ = new_ptr;
            size_ = size;
        }
    }

  private:
    void *ptr_;
    uint64_t size_;
};

} // namespace metaspore
