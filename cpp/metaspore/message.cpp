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

#include <metaspore/message.h>
#include <metaspore/stack_trace_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

void Message::AddTypedSlice(SmartArray<uint8_t> slice, DataType dataType) {
    AddSlice(std::move(slice));
    message_meta_.AddSliceDataType(dataType);
}

SmartArray<uint8_t> Message::GetSlice(size_t i) const {
    if (i >= slices_.size()) {
        std::string serr;
        serr.append("GetSlice failed as slice index is out of range. i: ");
        serr.append(std::to_string(i));
        serr.append(", slices_.size(): ");
        serr.append(std::to_string(slices_.size()));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    return slices_.at(i);
}

SmartArray<uint8_t> Message::GetTypedSlice(size_t i, DataType dataType) const {
    const std::vector<DataType> &sliceDataTypes = message_meta_.GetSliceDataTypes();
    if (i >= sliceDataTypes.size()) {
        std::string serr;
        serr.append("GetTypedSlice failed as slice index is out of range. i: ");
        serr.append(std::to_string(i));
        serr.append(", sliceDataTypes.size(): ");
        serr.append(std::to_string(sliceDataTypes.size()));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (i >= slices_.size()) {
        std::string serr;
        serr.append("GetTypedSlice failed as slice index is out of range. i: ");
        serr.append(std::to_string(i));
        serr.append(", slices_.size(): ");
        serr.append(std::to_string(slices_.size()));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (dataType != sliceDataTypes.at(i)) {
        std::string serr;
        serr.append("GetTypedSlice failed as data types mismatch. i: ");
        serr.append(std::to_string(i));
        serr.append(", dataType: ");
        serr.append(DataTypeToString(dataType));
        serr.append(", sliceDataTypes.at(i): ");
        serr.append(DataTypeToString(sliceDataTypes.at(i)));
        serr.append(".\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    return slices_.at(i);
}

std::string Message::ToString() const { return message_meta_.ToString(); }

std::string Message::ToJsonString() const { return to_json().dump(); }

json11::Json Message::to_json() const {
    std::vector<json11::Json> slices;
    slices.reserve(slices.size());
    for (auto &&slice : slices_)
        slices.push_back(slice.to_json());
    return json11::Json::object{
        {"message_meta", message_meta_},
        {"slices", std::move(slices)},
    };
}

} // namespace metaspore
