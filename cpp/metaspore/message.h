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

#include <metaspore/message_meta.h>
#include <metaspore/smart_array.h>

//
// ``message.h`` defines class ``Message`` which represents
// messages sent between Parameter Server nodes.
//
// ``Message`` consists of a metadata part and zero or more
// typed data slices.
//

namespace metaspore {

class Message {
  public:
    MessageMeta &GetMessageMeta() { return message_meta_; }
    const MessageMeta &GetMessageMeta() const { return message_meta_; }
    void SetMessageMeta(MessageMeta value) { message_meta_ = std::move(value); }

    const std::vector<SmartArray<uint8_t>> &GetSlices() const { return slices_; }
    void SetSlices(std::vector<SmartArray<uint8_t>> value) { slices_ = std::move(value); }
    void ClearSlices() { slices_.clear(); }
    void ClearSlicesAndDataTypes() {
        ClearSlices();
        message_meta_.ClearSliceDataTypes();
    }
    void AddSlice(SmartArray<uint8_t> value) { slices_.push_back(std::move(value)); }

    void AddTypedSlice(SmartArray<uint8_t> slice, DataType dataType);

    template <typename T> void AddTypedSlice(SmartArray<T> slice) {
        auto sa = slice.template Cast<uint8_t>();
        AddTypedSlice(std::move(sa), DataTypeToCode<T>::value);
    }

    SmartArray<uint8_t> GetSlice(size_t i) const;
    SmartArray<uint8_t> GetTypedSlice(size_t i, DataType dataType) const;

    template <typename T> SmartArray<T> GetTypedSlice(size_t i) const {
        SmartArray<uint8_t> slice = GetTypedSlice(i, DataTypeToCode<T>::value);
        return slice.Cast<T>();
    }

    std::shared_ptr<Message> Copy() const { return std::make_shared<Message>(*this); }

    std::string ToString() const;
    std::string ToJsonString() const;
    json11::Json to_json() const;

  private:
    MessageMeta message_meta_;
    std::vector<SmartArray<uint8_t>> slices_;
};

} // namespace metaspore
