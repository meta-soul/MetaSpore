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

# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import hf_preprocessor.hf_preprocessor_pb2 as hf__preprocessor__pb2


class HfPreprocessorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.HfTokenizer = channel.unary_unary(
                '/multimodal.service.HfPreprocessor/HfTokenizer',
                request_serializer=hf__preprocessor__pb2.HfTokenizerRequest.SerializeToString,
                response_deserializer=hf__preprocessor__pb2.HfTokenizerResponse.FromString,
                )
        self.HfTokenizerPush = channel.unary_unary(
                '/multimodal.service.HfPreprocessor/HfTokenizerPush',
                request_serializer=hf__preprocessor__pb2.HfTokenizerPushRequest.SerializeToString,
                response_deserializer=hf__preprocessor__pb2.HfTokenizerPushResponse.FromString,
                )


class HfPreprocessorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def HfTokenizer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def HfTokenizerPush(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_HfPreprocessorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'HfTokenizer': grpc.unary_unary_rpc_method_handler(
                    servicer.HfTokenizer,
                    request_deserializer=hf__preprocessor__pb2.HfTokenizerRequest.FromString,
                    response_serializer=hf__preprocessor__pb2.HfTokenizerResponse.SerializeToString,
            ),
            'HfTokenizerPush': grpc.unary_unary_rpc_method_handler(
                    servicer.HfTokenizerPush,
                    request_deserializer=hf__preprocessor__pb2.HfTokenizerPushRequest.FromString,
                    response_serializer=hf__preprocessor__pb2.HfTokenizerPushResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'multimodal.service.HfPreprocessor', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class HfPreprocessor(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def HfTokenizer(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/multimodal.service.HfPreprocessor/HfTokenizer',
            hf__preprocessor__pb2.HfTokenizerRequest.SerializeToString,
            hf__preprocessor__pb2.HfTokenizerResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def HfTokenizerPush(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/multimodal.service.HfPreprocessor/HfTokenizerPush',
            hf__preprocessor__pb2.HfTokenizerPushRequest.SerializeToString,
            hf__preprocessor__pb2.HfTokenizerPushResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
