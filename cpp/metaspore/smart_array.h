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

#include <common/hashmap/data_types.h>
#include <json11.hpp>
#include <memory>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/vector_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <stdint.h>
#include <vector>

//
// ``smart_array.h`` defines class template ``SmartArray`` which encapsulates
// a contiguous segment memory in ``std::shared_ptr`` so that data can be
// transferred among ``std::vector``, ``numpy.ndarray``, ``SmartArray`` and so on
// easily.
//

namespace metaspore {

template <typename T> class SmartArray {
  public:
    SmartArray() {}

    explicit SmartArray(size_t size) { Reset(size); }

    static SmartArray<T> Ref(const T *data, size_t size) {
        SmartArray<T> inst;
        inst.size_ = size;
        inst.capacity_ = size;
        inst.ptr_.reset(const_cast<T *>(data), [](T *data) {});
        return std::move(inst);
    }

    template <typename Deleter> static SmartArray<T> Create(T *data, size_t size, Deleter deleter) {
        SmartArray<T> inst;
        inst.size_ = size;
        inst.capacity_ = size;
        inst.ptr_.reset(data, deleter);
        return std::move(inst);
    }

    static SmartArray<T> Create(std::shared_ptr<T> data, size_t size) {
        SmartArray<T> inst;
        inst.size_ = size;
        inst.capacity_ = size;
        inst.ptr_ = data;
        return std::move(inst);
    }

    static SmartArray<T> Wrap(std::vector<T> vec) {
        T *data = nullptr;
        size_t size = 0;
        size_t capacity = 0;
        VectorDetachBuffer(vec, data, size, capacity);
        SmartArray<T> inst;
        inst.size_ = size;
        inst.capacity_ = capacity;
        inst.ptr_.reset(data, [size, capacity](T *data) {
            std::vector<T> vec;
            VectorAttachBuffer(vec, data, size, capacity);
        });
        return std::move(inst);
    }

    template <typename U> static SmartArray<T> Wrap(std::vector<U> vec) {
        return SmartArray<U>::Wrap(std::move(vec)).template Cast<T>();
    }

    template <typename U> SmartArray<U> Cast() const {
        const size_t n = size() * sizeof(T) / sizeof(U);
        if (size() * sizeof(T) != n * sizeof(U)) {
            const DataType t = DataTypeToCode<T>::value;
            const DataType u = DataTypeToCode<U>::value;
            std::string serr;
            serr.append(std::to_string(size()));
            serr.append(" ");
            serr.append(DataTypeToString(t));
            serr.append("(s) are not a multiple of ");
            serr.append(DataTypeToString(u));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        U *const u_ptr = reinterpret_cast<U *>(const_cast<T *>(data()));
        auto data = std::shared_ptr<U>(ptr(), u_ptr);
        return SmartArray<U>::Create(data, n);
    }

    SmartArray<T> Copy() const {
        std::vector<T> vec(data(), data() + size());
        return Wrap(std::move(vec));
    }

    void CopyFrom(const SmartArray<T> &other) {
        if (size() != other.size()) {
            std::string serr;
            serr.append("SmartArray lengths mismatch; ");
            serr.append(std::to_string(size()));
            serr.append(" != ");
            serr.append(std::to_string(other.size()));
            serr.append(".\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        memcpy(this->data(), other.data(), this->size() * sizeof(T));
    }

    void Reset(size_t size) {
        std::vector<T> vec(size);
        *this = Wrap(std::move(vec));
    }

    SmartArray<T> Slice(size_t begin, size_t end) const {
        if (begin > end) {
            std::string serr;
            serr.append("begin (");
            serr.append(std::to_string(begin));
            serr.append(") and end(");
            serr.append(std::to_string(end));
            serr.append(") doesn't form a range. size = ");
            serr.append(std::to_string(size()));
            serr.append("\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        if (end > size()) {
            std::string serr;
            serr.append("end (");
            serr.append(std::to_string(end));
            serr.append(") is out of range. begin = ");
            serr.append(std::to_string(begin));
            serr.append(", size = ");
            serr.append(std::to_string(size()));
            serr.append("\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        SmartArray<T> ret;
        ret.size_ = end - begin;
        ret.capacity_ = end - begin;
        ret.ptr_ = std::shared_ptr<T>(ptr_, ptr_.get() + begin);
        return ret;
    }

    bool empty() const { return size() == 0; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

    T *data() { return ptr_.get(); }
    T *begin() { return data(); }
    T *end() { return data() + size(); }

    const T *data() const { return ptr_.get(); }
    const T *begin() const { return data(); }
    const T *end() const { return data() + size(); }

    std::shared_ptr<T> &ptr() { return ptr_; }
    const std::shared_ptr<T> &ptr() const { return ptr_; }

    T &operator[](size_t i) { return data()[i]; }
    const T &operator[](size_t i) const { return data()[i]; }

    std::string ToString() const {
        std::string sout;
        sout.append(DataTypeToString(DataTypeToCode<T>::value));
        sout.append("[");
        sout.append(std::to_string(size()));
        sout.append("]: ");
        sout.append("[");
        for (size_t i = 0; i < size(); i++) {
            sout.append(i ? ", " : "");
            sout.append(std::to_string(as_number((*this)[i])));
        }
        sout.append("]");
        return sout;
    }

    std::string ToJsonString() const { return to_json().dump(); }

    json11::Json to_json() const {
        std::string sout;
        for (size_t i = 0; i < size(); i++) {
            sout.append(i ? "," : "");
            sout.append(std::to_string(as_number((*this)[i])));
        }
        return json11::Json::object{
            {"type", DataTypeToString(DataTypeToCode<T>::value)},
            {"size", static_cast<int>(size_)},
            {"capacity", static_cast<int>(capacity_)},
            {"data", sout},
        };
    }

  private:
    size_t size_ = 0;
    size_t capacity_ = 0;
    std::shared_ptr<T> ptr_;
};

} // namespace metaspore
