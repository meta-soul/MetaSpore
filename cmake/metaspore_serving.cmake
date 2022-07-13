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

find_package(range-v3 CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(mimalloc CONFIG REQUIRED)

target_link_libraries(arrow_static INTERFACE
    lz4::lz4
    utf8proc
    unofficial::brotli::brotlienc-static
    unofficial::brotli::brotlidec-static
    unofficial::brotli::brotlicommon-static
)

include("${CMAKE_CURRENT_LIST_DIR}/FindOnnxRuntimeCpuDefault.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/AsioGrpcProtobufGenerator.cmake")
if(ENABLE_GPU)
    include("${CMAKE_CURRENT_LIST_DIR}/FindOnnxRuntimeGpuDefault.cmake")
    find_package(CUDA)
endif()

set(SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/globals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/ort_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/inmem_sparse_lookup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/sparse_lookup_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/feature_compute_exec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/feature_compute_funcs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/converters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/tabular_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/sparse_feature_extraction_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/schema_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/dense_feature_extraction_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/grpc_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/grpc_server_shutdown.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/grpc_client_context_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/grpc_model_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/model_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/sparse_embedding_bag_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/py_preprocessing_process.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/py_preprocessing_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/py_preprocessing_ort_model.cpp
)

add_library(metaspore-serving STATIC
    ${SRCS}
)

target_include_directories(metaspore-serving PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

target_compile_options(metaspore-serving PUBLIC
    -funroll-loops
    -march=core-avx2
)

target_link_libraries(metaspore-serving PUBLIC
    metaspore-common
    Boost::filesystem
    Boost::system
    asio-grpc::asio-grpc
    fmt::fmt
    arrow_static
    range-v3
    xtensor
)

if(ENABLE_GPU)
    target_include_directories(metaspore-serving PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(metaspore-serving PUBLIC onnxruntime-gpu-default ${CUDA_LIBRARIES})
    target_compile_definitions(metaspore-serving PUBLIC ENABLE_GPU=1)
else()
    target_link_libraries(metaspore-serving PUBLIC onnxruntime-cpu-default)
    target_compile_definitions(metaspore-serving PUBLIC ENABLE_GPU=0)
endif()

add_executable(metaspore-serving-bin ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serving/main.cpp)
target_link_libraries(metaspore-serving-bin PRIVATE metaspore-serving mimalloc-static)

set_target_properties(metaspore-serving-bin PROPERTIES
        LINK_FLAGS "-Wl,-rpath,$ORIGIN/")

add_custom_command(TARGET metaspore-serving-bin
    POST_BUILD
    COMMAND ldd ${CMAKE_CURRENT_BINARY_DIR}/metaspore-serving-bin |
            egrep -v 'linux-vdso|ld-linux-x86-64|libpthread|libdl|libm|libc|librt' |
            cut -f 3 -d ' ' |
            xargs -L 1 -I so_file cp -n so_file ${CMAKE_CURRENT_BINARY_DIR}/
)

if(ENABLE_GPU)
    get_filename_component(ORT_GPU_LIB_DIR ${ORT_GPU_LIBRARY} DIRECTORY)
    add_custom_command(TARGET metaspore-serving-bin
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            libonnxruntime_providers_cuda.so
            libonnxruntime_providers_shared.so
            libonnxruntime_providers_tensorrt.so
            ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${ORT_GPU_LIB_DIR}
    )
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
message("Found Python at " ${Python_EXECUTABLE})

add_custom_command(TARGET metaspore-serving-bin
    POST_BUILD
    COMMAND ${Python_EXECUTABLE} -m grpc.tools.protoc
            -I=${CMAKE_CURRENT_SOURCE_DIR}/protos
            --python_out=${CMAKE_CURRENT_BINARY_DIR}
            --grpc_python_out=${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/protos/metaspore.proto
)

add_custom_command(TARGET metaspore-serving-bin
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/python/scripts/preprocessing/preprocessor_service.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/scripts/consul/consul_watch_load.py
            ${CMAKE_CURRENT_BINARY_DIR}
)

install(TARGETS metaspore-serving-bin RUNTIME DESTINATION bin)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/preprocessor_service.py
              ${CMAKE_CURRENT_BINARY_DIR}/metaspore_pb2_grpc.py
              ${CMAKE_CURRENT_BINARY_DIR}/metaspore_pb2.py
              DESTINATION bin)
