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

#include <metaspore/debug.h>
#include <metaspore/dense_tensor_partition.h>
#include <metaspore/file_utils.h>
#include <metaspore/io.h>
#include <metaspore/tensor_utils.h>
#include <string.h>

namespace metaspore {

void DenseTensorPartition::AllocateDataBlock(bool init) {
    size_t begin = 0;
    size_t end = 0;
    const int index = GetPartitionIndex();
    GetMeta().CheckDenseTensorMeta(index);
    GetMeta().ComputePartitionShapes(index, begin, end, &partition_data_shape_,
                                     &partition_state_shape_);
    const size_t item_size = DataTypeToSize(GetMeta().GetDataType());
    const size_t data_size = item_size * TotalElements(GetPartitionDataShape());
    data_.Reset(data_size);
    if (init) {
        DenseInitializer initializer = GetMeta().GetInitializer();
        if (!initializer)
            memset(data_.data(), 0, data_size);
        else
            initializer(GetMeta().GetName(), data_, GetMeta());
    }
    if (!GetPartitionStateShape().empty()) {
        const size_t state_size = item_size * TotalElements(GetPartitionStateShape());
        state_.Reset(state_size);
        if (init)
            memset(state_.data(), 0, state_size);
    }
}

void DenseTensorPartition::HandlePush(SmartArray<uint8_t> in, bool is_value, bool is_state) {
    DenseUpdater updater = GetMeta().GetUpdater();
    if (is_state)
        state_.CopyFrom(in);
    else if (!updater || is_value)
        data_.CopyFrom(in);
    else
        updater(GetMeta().GetName(), data_, in, state_, GetMeta());
}

SmartArray<uint8_t> DenseTensorPartition::HandlePull(bool is_state) {
    return is_state ? state_ : data_;
}

void DenseTensorPartition::HandlePushMeta(const DenseTensorMeta &meta) {
    if (!meta.IsCompatible(meta_)) {
        std::string serr;
        serr.append("Incompatible meta detected, can not update initializer and updater");
        serr.append(" of dense tensor '");
        serr.append(GetMeta().GetName());
        serr.append("'.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    meta_.SetInitializerByData(meta.GetInitializerAsData());
    meta_.SetUpdaterByData(meta.GetUpdaterAsData());
}

const DenseTensorMeta &DenseTensorPartition::HandlePullMeta() { return meta_; }

} // namespace metaspore
