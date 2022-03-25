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

#include <metaspore/message_meta.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/transport/TBufferTransports.h>

namespace metaspore {

std::string MessageMeta::ToString() const { return ToJsonString(); }

std::string MessageMeta::ToJsonString() const { return to_json().dump(); }

json11::Json MessageMeta::to_json() const {
    std::vector<json11::Json> slice_data_types;
    slice_data_types.reserve(slice_data_types_.size());
    for (auto &&type : slice_data_types_)
        slice_data_types.push_back(DataTypeToString(type));
    return json11::Json::object{
        {"message_id", message_id_},
        {"sender", sender_},
        {"receiver", receiver_},
        {"is_request", is_request_},
        {"is_exception", is_exception_},
        {"body", body_},
        {"slice_data_types", std::move(slice_data_types)},
        {"node_control", node_control_},
    };
}

TMessageMeta MessageMeta::PackAsThriftObject() const {
    TMessageMeta meta;
    meta.messageId = GetMessageId();
    meta.sender = GetSender();
    meta.receiver = GetReceiver();
    meta.isRequest = IsRequest();
    meta.isException = IsException();
    meta.body = GetBody();
    meta.sliceDataTypes.reserve(GetSliceDataTypes().size());
    for (DataType type : GetSliceDataTypes())
        meta.sliceDataTypes.push_back(static_cast<TDataType::type>(type));
    const NodeControl &control = GetNodeControl();
    meta.control.command = static_cast<TNodeControlCommand::type>(control.GetCommand());
    meta.control.nodes.reserve(control.GetNodes().size());
    for (const NodeInfo &n : control.GetNodes()) {
        meta.control.nodes.emplace_back();
        TNodeInfo &node = meta.control.nodes.back();
        node.role = static_cast<TNodeRole::type>(n.GetRole());
        node.nodeId = n.GetNodeId();
        node.hostName = n.GetHostName();
        node.port = n.GetPort();
    }
    meta.control.barrierGroup = control.GetBarrierGroup();
    return meta;
}

std::string MessageMeta::PackAsThriftJson() const {
    TMessageMeta meta = PackAsThriftObject();
    return apache::thrift::ThriftJSONString(meta);
}

SmartArray<uint8_t> MessageMeta::PackAsThriftBuffer() const {
    using apache::thrift::protocol::TBinaryProtocol;
    using apache::thrift::transport::TMemoryBuffer;
    std::shared_ptr<TMemoryBuffer> transport(new TMemoryBuffer());
    std::shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(transport));
    TMessageMeta meta = PackAsThriftObject();
    meta.write(protocol.get());
    uint8_t *buffer = nullptr;
    uint32_t size = 0;
    transport->getBuffer(&buffer, &size);
    return SmartArray<uint8_t>::Create(buffer, size, [transport](uint8_t *) {});
}

void MessageMeta::UnpackFromThriftObject(TMessageMeta &&meta) {
    SetMessageId(meta.messageId);
    SetSender(meta.sender);
    SetReceiver(meta.receiver);
    SetIsRequest(meta.isRequest);
    SetIsException(meta.isException);
    SetBody(std::move(meta.body));
    ClearSliceDataTypes();
    for (TDataType::type type : meta.sliceDataTypes)
        AddSliceDataType(static_cast<DataType>(type));
    NodeControl &control = GetNodeControl();
    control.SetCommand(static_cast<NodeControlCommand>(meta.control.command));
    control.ClearNodes();
    for (TNodeInfo &node : meta.control.nodes) {
        NodeInfo n;
        n.SetRole(static_cast<NodeRole>(node.role));
        n.SetNodeId(node.nodeId);
        n.SetHostName(std::move(node.hostName));
        n.SetPort(node.port);
        control.AddNode(std::move(n));
    }
    control.SetBarrierGroup(meta.control.barrierGroup);
}

void MessageMeta::UnpackFromThriftJson(const std::string &str) {
    using apache::thrift::protocol::TJSONProtocol;
    using apache::thrift::transport::TMemoryBuffer;
    const uint8_t *const ptr = reinterpret_cast<const uint8_t *>(str.data());
    const uint32_t size = static_cast<uint32_t>(str.size());
    std::shared_ptr<TMemoryBuffer> transport(new TMemoryBuffer(const_cast<uint8_t *>(ptr), size));
    std::shared_ptr<TJSONProtocol> protocol(new TJSONProtocol(transport));
    TMessageMeta meta;
    meta.read(protocol.get());
    UnpackFromThriftObject(std::move(meta));
}

void MessageMeta::UnpackFromThriftBuffer(const uint8_t *ptr, size_t size) {
    using apache::thrift::protocol::TBinaryProtocol;
    using apache::thrift::transport::TMemoryBuffer;
    std::shared_ptr<TMemoryBuffer> transport(
        new TMemoryBuffer(const_cast<uint8_t *>(ptr), static_cast<uint32_t>(size)));
    std::shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(transport));
    TMessageMeta meta;
    meta.read(protocol.get());
    UnpackFromThriftObject(std::move(meta));
}

void MessageMeta::UnpackFromThriftBuffer(const SmartArray<uint8_t> &buf) {
    UnpackFromThriftBuffer(buf.data(), buf.size());
}

void MessageMeta::UnpackFromThriftBuffer(const std::string_view &buf) {
    UnpackFromThriftBuffer(reinterpret_cast<const uint8_t *>(buf.data()), buf.size());
}

} // namespace metaspore
