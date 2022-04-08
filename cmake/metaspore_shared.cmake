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

get_project_version(project_version)
message(STATUS "project_version: ${project_version}")

find_package(Python REQUIRED COMPONENTS Interpreter Development)
message("Found Python at "${Python_EXECUTABLE})
find_package(Boost REQUIRED COMPONENTS)
find_package(PkgConfig REQUIRED)
find_package(pybind11 REQUIRED CONFIG)
find_package(AWSSDK REQUIRED CONFIG COMPONENTS s3)
find_package(ZLIB REQUIRED)

find_package(json11 CONFIG)

find_package(Thrift CONFIG)
if(NOT TARGET thrift::thrift)
    pkg_search_module(THRIFT REQUIRED IMPORTED_TARGET GLOBAL thrift)
    add_library(thrift::thrift ALIAS PkgConfig::THRIFT)
endif()

find_package(ZeroMQ CONFIG)
if(NOT TARGET libzmq-static)
    find_library(ZMQ_LIB zmq)
    if("${ZMQ_LIB}" STREQUAL "ZMQ_LIB-NOTFOUND")
        message(FATAL_ERROR "libzmq not found")
    endif()
    find_path(ZMQ_HEADER zmq.h)
    if("${ZMQ_HEADER}" STREQUAL "ZMQ_HEADER-NOTFOUND")
        message(FATAL_ERROR "zmq.h not found")
    endif()
    add_library(zmq::libzmq STATIC IMPORTED GLOBAL)
    set_target_properties(zmq::libzmq PROPERTIES
        IMPORTED_LOCATION "${ZMQ_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZMQ_HEADER}")
else()
    add_library(zmq::libzmq ALIAS libzmq-static)
endif()

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/gen/thrift/cpp/metaspore)

add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/gen/thrift/cpp/metaspore/message_meta_types.h
           ${PROJECT_BINARY_DIR}/gen/thrift/cpp/metaspore/message_meta_types.cpp
    COMMAND thrift -gen cpp:cob_style,moveable_types
            -out ${PROJECT_BINARY_DIR}/gen/thrift/cpp/metaspore
            ${PROJECT_SOURCE_DIR}/thrift/metaspore/message_meta.thrift
    DEPENDS ${PROJECT_SOURCE_DIR}/thrift/metaspore/message_meta.thrift)

add_library(metaspore_shared SHARED
    cpp/metaspore/stack_trace_utils.h
    cpp/metaspore/stack_trace_utils.cpp
    cpp/metaspore/thread_utils.h
    cpp/metaspore/thread_utils.cpp
    cpp/metaspore/string_utils.h
    cpp/metaspore/vector_utils.h
    cpp/metaspore/smart_array.h
    cpp/metaspore/memory_buffer.h
    cpp/metaspore/node_role.h
    cpp/metaspore/node_role.cpp
    cpp/metaspore/node_encoding.h
    cpp/metaspore/node_encoding.cpp
    cpp/metaspore/node_info.h
    cpp/metaspore/node_info.cpp
    cpp/metaspore/node_control_command.h
    cpp/metaspore/node_control_command.cpp
    cpp/metaspore/node_control.h
    cpp/metaspore/node_control.cpp
    cpp/metaspore/message_meta.h
    cpp/metaspore/message_meta.cpp
    cpp/metaspore/message.h
    cpp/metaspore/message.cpp
    cpp/metaspore/actor_config.cpp
    cpp/metaspore/message_transport.cpp
    cpp/metaspore/zeromq_transport.cpp
    cpp/metaspore/actor_process.cpp
    cpp/metaspore/node_manager.cpp
    cpp/metaspore/network_utils.cpp
    cpp/metaspore/ps_agent.cpp
    cpp/metaspore/ps_runner.cpp
    cpp/metaspore/io.cpp
    cpp/metaspore/filesys.cpp
    cpp/metaspore/local_filesys.cpp
    cpp/metaspore/s3_sdk_filesys.cpp
    ${PROJECT_BINARY_DIR}/gen/thrift/cpp/metaspore/message_meta_types.h
    ${PROJECT_BINARY_DIR}/gen/thrift/cpp/metaspore/message_meta_types.cpp
    cpp/metaspore/dense_tensor_meta.cpp
    cpp/metaspore/dense_tensor_partition.cpp
    cpp/metaspore/sparse_tensor_meta.cpp
    cpp/metaspore/sparse_tensor_partition.cpp
    cpp/metaspore/array_hash_map_reader.h
    cpp/metaspore/array_hash_map_writer.h
    cpp/metaspore/tensor_partition_store.cpp
    cpp/metaspore/dense_tensor.cpp
    cpp/metaspore/sparse_tensor.cpp
    cpp/metaspore/ps_default_agent.cpp
    cpp/metaspore/ps_helper.cpp
    cpp/metaspore/combine_schema.cpp
    cpp/metaspore/index_batch.cpp
    cpp/metaspore/model_metric_buffer.cpp
    cpp/metaspore/tensor_utils.cpp
    cpp/metaspore/pybind_utils.cpp
    cpp/metaspore/ms_ps_python_bindings.cpp
    cpp/metaspore/tensor_store_python_bindings.cpp
    cpp/metaspore/feature_extraction_python_bindings.cpp
)
set_target_properties(metaspore_shared PROPERTIES PREFIX "")
set_target_properties(metaspore_shared PROPERTIES OUTPUT_NAME _metaspore)
set_target_properties(metaspore_shared PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,-rpath,$ORIGIN/")
target_compile_definitions(metaspore_shared PRIVATE DMLC_USE_S3=1)
target_compile_definitions(metaspore_shared PRIVATE _METASPORE_VERSION="${project_version}")
target_compile_definitions(metaspore_shared PRIVATE DBG_MACRO_NO_WARNING)
target_include_directories(metaspore_shared PRIVATE ${PROJECT_SOURCE_DIR}/cpp)
target_include_directories(metaspore_shared PRIVATE ${PROJECT_BINARY_DIR}/gen/thrift/cpp)
target_link_libraries(metaspore_shared PRIVATE
    metaspore-common
    ${JSON11_LIBRARIES}
    pybind11::pybind11
    Python::Module
    aws-cpp-sdk-s3
    aws-cpp-sdk-core
    spdlog::spdlog
    Boost::headers
    thrift::thrift
    zmq::libzmq
)

add_custom_command(TARGET metaspore_shared
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    /lib/x86_64-linux-gnu/libstdc++.so.6 ${CMAKE_BINARY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy
    /lib/x86_64-linux-gnu/libgcc_s.so.1 ${CMAKE_BINARY_DIR}/
)