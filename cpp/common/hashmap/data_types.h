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
#include <string.h>
#include <string>

namespace metaspore {

#define MS_DATA_STRUCTURES_INTEGRAL_DATA_TYPES(X)                                                  \
    X(int8_t, int8, Int8)                                                                          \
    X(int16_t, int16, Int16)                                                                       \
    X(int32_t, int32, Int32)                                                                       \
    X(int64_t, int64, Int64)                                                                       \
    X(uint8_t, uint8, UInt8)                                                                       \
    X(uint16_t, uint16, UInt16)                                                                    \
    X(uint32_t, uint32, UInt32)                                                                    \
    X(uint64_t, uint64, UInt64)                                                                    \
    /**/

#define MS_DATA_STRUCTURES_FLOATING_DATA_TYPES(X)                                                  \
    X(float, float32, Float32)                                                                     \
    X(double, float64, Float64)                                                                    \
    /**/

#define MS_DATA_STRUCTURES_DATA_TYPES(X)                                                           \
    MS_DATA_STRUCTURES_INTEGRAL_DATA_TYPES(X)                                                      \
    MS_DATA_STRUCTURES_FLOATING_DATA_TYPES(X)                                                      \
    /**/

enum class DataType : uint8_t {
#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u) u,
    MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)
};

constexpr DataType NullDataType = static_cast<DataType>(-1);
constexpr DataType InvalidDataType = static_cast<DataType>(-2);
constexpr const char *NullDataTypeString = "null";
constexpr const char *InvalidDataTypeString = "invalid";

template <typename T> struct DataTypeToCode;

#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u)                                                  \
    template <> struct DataTypeToCode<t> { static constexpr DataType value = DataType::u; };       \
/**/
MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)

constexpr size_t DataTypeToSize(DataType type) {
    switch (type) {
#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u)                                                  \
    case DataType::u:                                                                              \
        return sizeof(t);
        MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)
    default:
        return 0;
    }
}

constexpr const char *DataTypeToString(DataType type) {
    switch (type) {
#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u)                                                  \
    case DataType::u:                                                                              \
        return #l;
        MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)
    default:
        return InvalidDataTypeString;
    }
}

inline DataType DataTypeFromCString(const char *str) {
#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u)                                                  \
    if (strcmp(str, #l) == 0)                                                                      \
        return DataType::u;
    MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)
    return InvalidDataType;
}

inline DataType DataTypeFromString(const std::string &str) {
    return DataTypeFromCString(str.c_str());
}

constexpr const char *NullableDataTypeToString(DataType type) {
    switch (type) {
#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u)                                                  \
    case DataType::u:                                                                              \
        return #l;
        MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)
    default:
        return (type == NullDataType) ? NullDataTypeString : InvalidDataTypeString;
    }
}

inline DataType NullableDataTypeFromCString(const char *str) {
#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u)                                                  \
    if (strcmp(str, #l) == 0)                                                                      \
        return DataType::u;
    MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)
    if (strcmp(str, NullDataTypeString) == 0)
        return NullDataType;
    return InvalidDataType;
}

inline DataType NullableDataTypeFromString(const std::string &str) {
    return NullableDataTypeFromCString(str.c_str());
}

template <typename T> inline T as_number(T value) { return value; }

inline int32_t as_number(int8_t value) { return static_cast<int32_t>(value); }

inline uint32_t as_number(uint8_t value) { return static_cast<uint32_t>(value); }

} // namespace metaspore