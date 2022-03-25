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
#include <metaspore/message_meta_types.h>
#include <metaspore/node_control.h>
#include <metaspore/smart_array.h>

//
// ``message_meta.h`` defines class ``MessageMeta`` which stores
// metadata of messages sent between Parameter Server nodes.
//

namespace metaspore {

class MessageMeta {
  public:
    int GetMessageId() const { return message_id_; }
    void SetMessageId(int value) { message_id_ = value; }

    int GetSender() const { return sender_; }
    void SetSender(int value) { sender_ = value; }

    int GetReceiver() const { return receiver_; }
    void SetReceiver(int value) { receiver_ = value; }

    bool IsRequest() const { return is_request_; }
    void SetIsRequest(bool value) { is_request_ = value; }

    bool IsException() const { return is_exception_; }
    void SetIsException(bool value) { is_exception_ = value; }

    const std::string &GetBody() const { return body_; }
    void SetBody(std::string value) { body_ = std::move(value); }

    const std::vector<DataType> &GetSliceDataTypes() const { return slice_data_types_; }
    void SetSliceDataTypes(std::vector<DataType> value) { slice_data_types_ = std::move(value); }
    void ClearSliceDataTypes() { slice_data_types_.clear(); }
    void AddSliceDataType(DataType value) { slice_data_types_.push_back(value); }

    NodeControl &GetNodeControl() { return node_control_; }
    const NodeControl &GetNodeControl() const { return node_control_; }
    void SetNodeControl(NodeControl value) { node_control_ = std::move(value); }

    std::string ToString() const;
    std::string ToJsonString() const;
    json11::Json to_json() const;

    TMessageMeta PackAsThriftObject() const;
    std::string PackAsThriftJson() const;
    SmartArray<uint8_t> PackAsThriftBuffer() const;

    void UnpackFromThriftObject(TMessageMeta &&meta);
    void UnpackFromThriftJson(const std::string &str);
    void UnpackFromThriftBuffer(const uint8_t *ptr, size_t size);
    void UnpackFromThriftBuffer(const SmartArray<uint8_t> &buf);
    void UnpackFromThriftBuffer(const std::string_view &buf);

  private:
    int message_id_ = -1;
    int sender_ = -1;
    int receiver_ = -1;
    bool is_request_ = false;
    bool is_exception_ = false;
    std::string body_;
    std::vector<DataType> slice_data_types_;
    NodeControl node_control_;
};

} // namespace metaspore
