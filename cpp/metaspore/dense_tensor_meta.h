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
#include <common/hash_utils.h>
#include <common/hashmap/data_types.h>
#include <functional>
#include <json11.hpp>
#include <metaspore/smart_array.h>
#include <metaspore/string_utils.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace metaspore {

using DenseInitializer = std::function<void(const std::string &name, SmartArray<uint8_t> data,
                                            const class DenseTensorMeta &meta)>;

using DenseUpdater =
    std::function<void(const std::string &name, SmartArray<uint8_t> param, SmartArray<uint8_t> grad,
                       SmartArray<uint8_t> state, const class DenseTensorMeta &meta)>;

class DenseTensorMeta {
  public:
    const std::string &GetName() const { return name_; }
    void SetName(std::string value) { name_ = std::move(value); }

    DataType GetDataType() const { return data_type_; }
    void SetDataType(DataType value) { data_type_ = value; }

    const std::vector<size_t> &GetDataShape() const { return data_shape_; }
    void SetDataShape(std::vector<size_t> value) { data_shape_ = std::move(value); }

    const std::vector<size_t> &GetStateShape() const { return state_shape_; }
    void SetStateShape(std::vector<size_t> value) { state_shape_ = std::move(value); }

    DenseInitializer GetInitializer() const { return initializer_; }
    void SetInitializer(DenseInitializer value) { initializer_ = std::move(value); }

    DenseUpdater GetUpdater() const { return updater_; }
    void SetUpdater(DenseUpdater value) { updater_ = std::move(value); }

    int GetPartitionCount() const { return partition_count_; }
    void SetPartitionCount(int value) { partition_count_ = value; }

    void CheckDenseTensorMeta(int index) const;

    void ComputePartitionShapesWithHash(size_t hash, int index, size_t &begin, size_t &end,
                                        std::vector<size_t> *partition_data_shape,
                                        std::vector<size_t> *partition_state_shape) const;

    void ComputePartitionShapes(int index, size_t &begin, size_t &end,
                                std::vector<size_t> *partition_data_shape,
                                std::vector<size_t> *partition_state_shape) const;

    size_t GetNameHash() const { return BKDRHash(name_); }

    void SetInitializerByData(std::string data);
    void SetUpdaterByData(std::string data);

    std::string GetInitializerAsData() const;
    std::string GetUpdaterAsData() const;

    std::string ToString() const;
    std::string ToJsonString() const;
    json11::Json to_json() const;

    static DenseTensorMeta FromJsonString(const std::string &str);
    static DenseTensorMeta FromJson(json11::Json json);

    bool IsCompatible(const DenseTensorMeta &rhs) const;

    bool operator==(const DenseTensorMeta &rhs) const;
    bool operator!=(const DenseTensorMeta &rhs) const { return !(*this == rhs); }

  private:
    std::string name_;
    DataType data_type_ = NullDataType;
    std::vector<size_t> data_shape_;
    std::vector<size_t> state_shape_;
    DenseInitializer initializer_;
    DenseUpdater updater_;
    std::any initializer_object_;
    std::any updater_object_;
    int partition_count_ = -1;
};

} // namespace metaspore
