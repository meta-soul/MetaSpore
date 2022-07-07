#
# Copyright 2022 DMetaSoul
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: metaspore.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fmetaspore.proto\x12\x11metaspore.serving\"\x8f\x02\n\x0ePredictRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x45\n\nparameters\x18\x03 \x03(\x0b\x32\x31.metaspore.serving.PredictRequest.ParametersEntry\x12?\n\x07payload\x18\x05 \x03(\x0b\x32..metaspore.serving.PredictRequest.PayloadEntry\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a.\n\x0cPayloadEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\"\xe9\x01\n\x0cPredictReply\x12=\n\x07payload\x18\x01 \x03(\x0b\x32,.metaspore.serving.PredictReply.PayloadEntry\x12;\n\x06\x65xtras\x18\x03 \x03(\x0b\x32+.metaspore.serving.PredictReply.ExtrasEntry\x1a.\n\x0cPayloadEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\x1a-\n\x0b\x45xtrasEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"D\n\x0bLoadRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x03 \x01(\t\x12\x10\n\x08\x64ir_path\x18\x05 \x01(\t\"\x18\n\tLoadReply\x12\x0b\n\x03msg\x18\x01 \x01(\t2\xa2\x01\n\x07Predict\x12O\n\x07Predict\x12!.metaspore.serving.PredictRequest\x1a\x1f.metaspore.serving.PredictReply\"\x00\x12\x46\n\x04Load\x12\x1e.metaspore.serving.LoadRequest\x1a\x1c.metaspore.serving.LoadReply\"\x00\x32N\n\x04Load\x12\x46\n\x04Load\x12\x1e.metaspore.serving.LoadRequest\x1a\x1c.metaspore.serving.LoadReply\"\x00\x42\x45\n\x1f\x63om.dmetasoul.metaspore.servingB\x15MetaSporeServingProtoP\x01\xf8\x01\x01\xa2\x02\x05SPOREb\x06proto3')



_PREDICTREQUEST = DESCRIPTOR.message_types_by_name['PredictRequest']
_PREDICTREQUEST_PARAMETERSENTRY = _PREDICTREQUEST.nested_types_by_name['ParametersEntry']
_PREDICTREQUEST_PAYLOADENTRY = _PREDICTREQUEST.nested_types_by_name['PayloadEntry']
_PREDICTREPLY = DESCRIPTOR.message_types_by_name['PredictReply']
_PREDICTREPLY_PAYLOADENTRY = _PREDICTREPLY.nested_types_by_name['PayloadEntry']
_PREDICTREPLY_EXTRASENTRY = _PREDICTREPLY.nested_types_by_name['ExtrasEntry']
_LOADREQUEST = DESCRIPTOR.message_types_by_name['LoadRequest']
_LOADREPLY = DESCRIPTOR.message_types_by_name['LoadReply']
PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), {

  'ParametersEntry' : _reflection.GeneratedProtocolMessageType('ParametersEntry', (_message.Message,), {
    'DESCRIPTOR' : _PREDICTREQUEST_PARAMETERSENTRY,
    '__module__' : 'metaspore_pb2'
    # @@protoc_insertion_point(class_scope:metaspore.serving.PredictRequest.ParametersEntry)
    })
  ,

  'PayloadEntry' : _reflection.GeneratedProtocolMessageType('PayloadEntry', (_message.Message,), {
    'DESCRIPTOR' : _PREDICTREQUEST_PAYLOADENTRY,
    '__module__' : 'metaspore_pb2'
    # @@protoc_insertion_point(class_scope:metaspore.serving.PredictRequest.PayloadEntry)
    })
  ,
  'DESCRIPTOR' : _PREDICTREQUEST,
  '__module__' : 'metaspore_pb2'
  # @@protoc_insertion_point(class_scope:metaspore.serving.PredictRequest)
  })
_sym_db.RegisterMessage(PredictRequest)
_sym_db.RegisterMessage(PredictRequest.ParametersEntry)
_sym_db.RegisterMessage(PredictRequest.PayloadEntry)

PredictReply = _reflection.GeneratedProtocolMessageType('PredictReply', (_message.Message,), {

  'PayloadEntry' : _reflection.GeneratedProtocolMessageType('PayloadEntry', (_message.Message,), {
    'DESCRIPTOR' : _PREDICTREPLY_PAYLOADENTRY,
    '__module__' : 'metaspore_pb2'
    # @@protoc_insertion_point(class_scope:metaspore.serving.PredictReply.PayloadEntry)
    })
  ,

  'ExtrasEntry' : _reflection.GeneratedProtocolMessageType('ExtrasEntry', (_message.Message,), {
    'DESCRIPTOR' : _PREDICTREPLY_EXTRASENTRY,
    '__module__' : 'metaspore_pb2'
    # @@protoc_insertion_point(class_scope:metaspore.serving.PredictReply.ExtrasEntry)
    })
  ,
  'DESCRIPTOR' : _PREDICTREPLY,
  '__module__' : 'metaspore_pb2'
  # @@protoc_insertion_point(class_scope:metaspore.serving.PredictReply)
  })
_sym_db.RegisterMessage(PredictReply)
_sym_db.RegisterMessage(PredictReply.PayloadEntry)
_sym_db.RegisterMessage(PredictReply.ExtrasEntry)

LoadRequest = _reflection.GeneratedProtocolMessageType('LoadRequest', (_message.Message,), {
  'DESCRIPTOR' : _LOADREQUEST,
  '__module__' : 'metaspore_pb2'
  # @@protoc_insertion_point(class_scope:metaspore.serving.LoadRequest)
  })
_sym_db.RegisterMessage(LoadRequest)

LoadReply = _reflection.GeneratedProtocolMessageType('LoadReply', (_message.Message,), {
  'DESCRIPTOR' : _LOADREPLY,
  '__module__' : 'metaspore_pb2'
  # @@protoc_insertion_point(class_scope:metaspore.serving.LoadReply)
  })
_sym_db.RegisterMessage(LoadReply)

_PREDICT = DESCRIPTOR.services_by_name['Predict']
_LOAD = DESCRIPTOR.services_by_name['Load']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\037com.dmetasoul.metaspore.servingB\025MetaSporeServingProtoP\001\370\001\001\242\002\005SPORE'
  _PREDICTREQUEST_PARAMETERSENTRY._options = None
  _PREDICTREQUEST_PARAMETERSENTRY._serialized_options = b'8\001'
  _PREDICTREQUEST_PAYLOADENTRY._options = None
  _PREDICTREQUEST_PAYLOADENTRY._serialized_options = b'8\001'
  _PREDICTREPLY_PAYLOADENTRY._options = None
  _PREDICTREPLY_PAYLOADENTRY._serialized_options = b'8\001'
  _PREDICTREPLY_EXTRASENTRY._options = None
  _PREDICTREPLY_EXTRASENTRY._serialized_options = b'8\001'
  _PREDICTREQUEST._serialized_start=39
  _PREDICTREQUEST._serialized_end=310
  _PREDICTREQUEST_PARAMETERSENTRY._serialized_start=213
  _PREDICTREQUEST_PARAMETERSENTRY._serialized_end=262
  _PREDICTREQUEST_PAYLOADENTRY._serialized_start=264
  _PREDICTREQUEST_PAYLOADENTRY._serialized_end=310
  _PREDICTREPLY._serialized_start=313
  _PREDICTREPLY._serialized_end=546
  _PREDICTREPLY_PAYLOADENTRY._serialized_start=264
  _PREDICTREPLY_PAYLOADENTRY._serialized_end=310
  _PREDICTREPLY_EXTRASENTRY._serialized_start=501
  _PREDICTREPLY_EXTRASENTRY._serialized_end=546
  _LOADREQUEST._serialized_start=548
  _LOADREQUEST._serialized_end=616
  _LOADREPLY._serialized_start=618
  _LOADREPLY._serialized_end=642
  _PREDICT._serialized_start=645
  _PREDICT._serialized_end=807
  _LOAD._serialized_start=809
  _LOAD._serialized_end=887
# @@protoc_insertion_point(module_scope)
