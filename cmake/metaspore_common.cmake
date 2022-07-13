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

find_package(gflags CONFIG REQUIRED)
find_package(asio-grpc CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)

set(SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/globals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/hashtable_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/hash_uniquifier.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/map_file_header.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/memory_usage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/memory_mapped_array_hash_map_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/arrow/arrow_record_batch_serde.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/arrow/arrow_tensor_serde.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/features/feature_compute_funcs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/features/feature_compute_exec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/features/schema_parser.cpp
)

set(PROTOS
    ${CMAKE_CURRENT_SOURCE_DIR}/protos/metaspore.proto
)

set(PROTO_INC_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen/proto/cpp)
set(PROTO_SRC_DIR ${PROTO_INC_DIR}/common)
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})

asio_grpc_protobuf_generate(
    GENERATE_GRPC
    OUT_VAR PROTO_SRCS
    OUT_DIR "${PROTO_SRC_DIR}"
    PROTOS ${PROTOS})

add_library(metaspore-common STATIC
    ${SRCS}
    ${PROTO_SRCS}
)

target_compile_options(metaspore-common PRIVATE
    -funroll-loops
    -march=core-avx2
)

target_include_directories(metaspore-common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${PROTO_INC_DIR}
)

target_link_libraries(metaspore-common PUBLIC
    gflags
    xtensor
)
