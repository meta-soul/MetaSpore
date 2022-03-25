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

#include <any>
#include <common/hashmap/data_types.h>
#include <functional>
#include <json11.hpp>
#include <metaspore/smart_array.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace metaspore {

using SparseInitializer =
    std::function<void(const std::string &name, SmartArray<uint8_t> data, SmartArray<uint8_t> keys,
                       const class SparseTensorMeta &meta)>;

using SparseUpdater = std::function<void(
    const std::string &name, SmartArray<uint8_t> param, SmartArray<uint8_t> grad,
    SmartArray<uint8_t> indices, SmartArray<uint8_t> keys, const class SparseTensorMeta &meta)>;

class SparseTensorMeta {
  public:
    const std::string &GetName() const { return name_; }
    void SetName(std::string value) { name_ = std::move(value); }

    DataType GetDataType() const { return data_type_; }
    void SetDataType(DataType value) { data_type_ = value; }

    const std::vector<size_t> &GetSliceDataShape() const { return slice_data_shape_; }
    void SetSliceDataShape(std::vector<size_t> value) { slice_data_shape_ = std::move(value); }

    const std::vector<size_t> &GetSliceStateShape() const { return slice_state_shape_; }
    void SetSliceStateShape(std::vector<size_t> value) { slice_state_shape_ = std::move(value); }

    SparseInitializer GetInitializer() const { return initializer_; }
    void SetInitializer(SparseInitializer value) { initializer_ = std::move(value); }

    SparseUpdater GetUpdater() const { return updater_; }
    void SetUpdater(SparseUpdater value) { updater_ = std::move(value); }

    int GetPartitionCount() const { return partition_count_; }
    void SetPartitionCount(int value) { partition_count_ = value; }

    void CheckSparseTensorMeta(int index) const;
    void ComputeSliceInfo();

    size_t GetSliceDataLength() const { return slice_data_length_; }
    size_t GetSliceStateLength() const { return slice_state_length_; }
    size_t GetSliceAgeOffset() const { return slice_age_offset_; }
    size_t GetSliceTotalBytes() const { return slice_total_bytes_; }
    size_t GetSliceDataStateBytes() const { return GetSliceAgeOffset(); }

    void SetInitializerByData(std::string data);
    void SetUpdaterByData(std::string data);

    std::string GetInitializerAsData() const;
    std::string GetUpdaterAsData() const;

    std::string ToString() const;
    std::string ToJsonString() const;
    json11::Json to_json() const;

    static SparseTensorMeta FromJsonString(const std::string &str);
    static SparseTensorMeta FromJson(json11::Json json);

    bool IsCompatible(const SparseTensorMeta &rhs) const;
    bool IsCompatibleRelaxed(const SparseTensorMeta &rhs, bool data_only) const;

    bool operator==(const SparseTensorMeta &rhs) const;
    bool operator!=(const SparseTensorMeta &rhs) const { return !(*this == rhs); }

  private:
    std::string name_;
    DataType data_type_ = NullDataType;
    std::vector<size_t> slice_data_shape_;
    std::vector<size_t> slice_state_shape_;
    SparseInitializer initializer_;
    SparseUpdater updater_;
    std::any initializer_object_;
    std::any updater_object_;
    int partition_count_ = -1;
    size_t slice_data_length_ = size_t(-1);
    size_t slice_state_length_ = size_t(-1);
    size_t slice_age_offset_ = size_t(-1);
    size_t slice_total_bytes_ = size_t(-1);
};

} // namespace metaspore
