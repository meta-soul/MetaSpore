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

//
// ``message_meta.thrift`` defines the Thrift representation of ``MessageMeta``.
//
// This is used to serialize ``MessageMeta`` to and from byte buffer. Currently,
// only the serialization mechanism of Thrift is used.
//

namespace cpp metaspore

enum TNodeRole
{
    Null = -1;
    Coordinator = 0;
    Server = 1;
    Worker = 2;
}

struct TNodeInfo
{
    1: required TNodeRole role;
    2: required i32 nodeId;
    3: required string hostName;
    4: required i32 port;
}

enum TNodeControlCommand
{
    Null = -1;
    Terminate = 0;
    AddNode = 1;
    Barrier = 2;
}

struct TNodeControl
{
    1: required TNodeControlCommand command;
    2: required list<TNodeInfo> nodes;
    3: required i32 barrierGroup;
}

enum TDataType
{
    Null = -1;
    Int8 = 0;
    Int16 = 1;
    Int32 = 2;
    Int64 = 3;
    UInt8 = 4;
    UInt16 = 5;
    UInt32 = 6;
    UInt64 = 7;
    Float32 = 8;
    Float64 = 9;
}

struct TMessageMeta
{
    1: required i32 messageId;
    2: required i32 sender;
    3: required i32 receiver;
    4: required bool isRequest;
    5: required bool isException;
    6: required string body;
    7: required list<TDataType> sliceDataTypes;
    8: required TNodeControl control;
}
