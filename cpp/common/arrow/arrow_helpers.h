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

#include <arrow/api.h>
#include <common/types.h>
#include <common/utils.h>

namespace metaspore {

class ArrowHelpers {
  public:
    template <typename TypeClass, typename ValueType = typename TypeClass::c_type>
    static arrow::Result<std::shared_ptr<arrow::NumericTensor<TypeClass>>>
    create_1d_tensor(std::initializer_list<ValueType> l) {
        ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateBuffer(l.size() * sizeof(ValueType)));
        VectorAssign<ValueType, 0>::assign(std::data(l), (ValueType *)buffer->mutable_data(),
                                           l.size());
        return arrow::NumericTensor<TypeClass>::Make(
            std::shared_ptr<arrow::Buffer>(buffer.release()),
            std::vector<int64_t>{(int64_t)l.size()});
    }

    template <typename TypeClass, typename ValueType = typename TypeClass::c_type>
    static arrow::Result<std::shared_ptr<arrow::NumericTensor<TypeClass>>>
    create_2d_tensor(std::initializer_list<std::initializer_list<ValueType>> l) {
        ARROW_ASSIGN_OR_RAISE(
            auto buffer, arrow::AllocateBuffer(l.size() * l.begin()->size() * sizeof(ValueType)));
        size_t i = 0;
        for (const auto &outer : l) {
            size_t cols = outer.size();
            VectorAssign<ValueType, 0>::assign(
                std::data(outer), ((ValueType *)buffer->mutable_data()) + i * cols, cols);
        }
        return arrow::NumericTensor<TypeClass>::Make(
            std::shared_ptr<arrow::Buffer>(buffer.release()),
            std::vector<int64_t>{(int64_t)l.size(), (int64_t)l.begin()->size()});
    }

    template <typename TYPE,
              typename = typename std::enable_if<arrow::is_number_type<TYPE>::value |
                                                 arrow::is_boolean_type<TYPE>::value |
                                                 arrow::is_temporal_type<TYPE>::value>::type>
    static arrow::Result<std::shared_ptr<arrow::Array>>
    GetArrayDataSample(const std::vector<typename TYPE::c_type> &values) {
        using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<TYPE>::ArrayType;
        using ARROW_BUILDER_TYPE = typename arrow::TypeTraits<TYPE>::BuilderType;
        ARROW_BUILDER_TYPE builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(values.size()));
        std::shared_ptr<ARROW_ARRAY_TYPE> array;
        ARROW_RETURN_NOT_OK(builder.AppendValues(values));
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        return array;
    }

    template <class TYPE>
    static arrow::Result<std::shared_ptr<arrow::Array>>
    GetBinaryArrayDataSample(const std::vector<std::string> &values) {
        using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<TYPE>::ArrayType;
        using ARROW_BUILDER_TYPE = typename arrow::TypeTraits<TYPE>::BuilderType;
        ARROW_BUILDER_TYPE builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(values.size()));
        std::shared_ptr<ARROW_ARRAY_TYPE> array;
        ARROW_RETURN_NOT_OK(builder.AppendValues(values));
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        return array;
    }

    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    GetSampleRecordBatch(const arrow::ArrayVector array_vector,
                         const arrow::FieldVector &field_vector) {
        std::shared_ptr<arrow::RecordBatch> record_batch;
        ARROW_ASSIGN_OR_RAISE(auto struct_result,
                              arrow::StructArray::Make(array_vector, field_vector));
        return record_batch->FromStructArray(struct_result);
    }

    template <typename ArrayType, typename TypeClass = typename ArrayType::TypeClass,
              typename ValueType = typename TypeClass::c_type>
    static void print_list_array(std::shared_ptr<arrow::ListArray> list_array) {
        auto offset_array = std::dynamic_pointer_cast<arrow::Int32Array>(list_array->offsets());
        auto value_array = std::dynamic_pointer_cast<ArrayType>(list_array->values());
        const auto offsets = offset_array->raw_values();
        for (int64_t i = 0; i < offset_array->length() - 1; ++i) {
            if (list_array->IsNull(i)) {
                fmt::print("[]\n, ");
            } else {
                int32_t begin = offsets[i];
                int32_t end = offsets[i + 1];
                fmt::print("[");
                for (int j = begin; j < end; ++j) {
                    fmt::print("{}, ", value_array->Value(j));
                }
                fmt::print("]\n");
            }
        }
    }
};

struct HashListAccessor {
    int64_t _begin;
    int64_t _end;
    std::shared_ptr<arrow::UInt64Array> array;

    size_t size() const { return _end - _begin; }

    bool empty() const { return size() == 0UL; }

    uint64_t operator[](size_t i) const { return *(array->raw_values() + _begin + i); }

    auto begin() const { return array->raw_values() + _begin; }

    auto end() const { return array->raw_values() + _end; }

    template <template <typename> typename Container>
    static arrow::Result<Container<std::function<HashListAccessor(int64_t i)>>>
    create_accessor_makers(const Container<std::shared_ptr<arrow::Array>> &arrays) {
        using MakeAccessorFunc = std::function<HashListAccessor(int64_t i)>;
        Container<MakeAccessorFunc> accessor_maker;
        accessor_maker.reserve(arrays.size());
        for (const auto &array : arrays) {
            if (auto uint64_array = std::dynamic_pointer_cast<arrow::UInt64Array>(array);
                uint64_array) {
                accessor_maker.emplace_back([=](int64_t i) {
                    if (uint64_array->IsNull(i)) {
                        return HashListAccessor{._begin = 0, ._end = 0, .array = nullptr};
                    } else {
                        return HashListAccessor{._begin = i, ._end = i + 1, .array = uint64_array};
                    }
                });
            } else if (auto uint64_list_array = std::dynamic_pointer_cast<arrow::ListArray>(array);
                       uint64_list_array &&
                       uint64_list_array->list_type()->value_type()->Equals(arrow::uint64())) {
                auto offset_array =
                    std::dynamic_pointer_cast<arrow::Int32Array>(uint64_list_array->offsets());
                if (!offset_array) {
                    return arrow::Status::Invalid(
                        "BKDRHashCombineKernelListUInt64: input's offset is not an int32 array");
                }
                accessor_maker.emplace_back([=](int64_t i) {
                    if (uint64_list_array->IsNull(i)) {
                        return HashListAccessor{._begin = 0, ._end = 0, .array = nullptr};
                    } else {
                        return HashListAccessor{._begin = offset_array->GetView(i),
                                                ._end = offset_array->GetView(i + 1),
                                                .array =
                                                    std::static_pointer_cast<arrow::UInt64Array>(
                                                        uint64_list_array->values())};
                    }
                });
            } else {
                return arrow::Status::Invalid(
                    "BKDRHashCombineKernelListUInt64 only accepts array of "
                    "uint64 or array of list<uint64>, but got " +
                    array->type()->ToString());
            }
        }
        return accessor_maker;
    }
};

} // namespace metaspore
