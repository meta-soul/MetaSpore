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

find_package(GTest CONFIG REQUIRED) 
enable_testing()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/data/MNIST/raw/t10k-images-idx3-ubyte
    COMMAND find . -type f -name '*.gz' -exec sh -c '[ -e \$\${1%.gz} ] || gunzip -k \$\$1' find-sh {} \\\;
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/data/MNIST/raw
)
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/data
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/cpp/tests/data ${CMAKE_BINARY_DIR}/data)
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/schema
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_SOURCE_DIR}/cpp/tests/schema ${CMAKE_BINARY_DIR}/schema)
add_custom_target(copy_files ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/data/MNIST/raw/t10k-images-idx3-ubyte
            ${CMAKE_BINARY_DIR}/data
            ${CMAKE_BINARY_DIR}/schema
)

if(BUILD_SERVING_BIN)
    add_library(metaspore-test IMPORTED INTERFACE)
    target_include_directories(metaspore-test INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests)
    target_link_libraries(metaspore-test INTERFACE metaspore-serving GTest::gtest)

    add_dependencies(metaspore-test copy_files)

    function(add_cpp_test test_name file_name)
        add_executable(${test_name}
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/${file_name}
        )
        target_link_libraries(${test_name} PRIVATE
            metaspore-test
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

    add_cpp_test(test_threadpool serving/threadpool_test.cpp)
    add_cpp_test(test_ort_model serving/ort_model_test.cpp)
    add_cpp_test(test_sparse_lookup_model serving/sparse_lookup_model_test.cpp)
    add_cpp_test(test_feature_compute_exec serving/feature_compute_exec_test.cpp)
    add_cpp_test(test_feature_compute_funcs serving/feature_compute_funcs_test.cpp)
    add_cpp_test(test_arrow_plan serving/arrow_plan_test.cpp)
    add_cpp_test(test_schema_parser serving/schema_parser_test.cpp)
    add_cpp_test(test_sparse_ctr_model serving/sparse_ctr_model_test.cpp)
    add_cpp_test(test_tabular_xgboost_model serving/tabular_xgboost_test.cpp)

    add_executable(map-dumper-float
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/mdumper-float.cpp
    )
    target_link_libraries(map-dumper-float PRIVATE
        metaspore-serving
    )
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CreateVirtualEnvironment.cmake)

CreateVirtualEnvironment(testing_venv
    REQUIREMENTS_TXT ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/requirements.txt
    OUT_PYTHON_EXE PYTHON_EXE
    OUT_BINARY_DIR PYTHON_BIN_DIR)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/metaspore_pb2.py ${CMAKE_BINARY_DIR}/metaspore_pb2_grpc.py
    COMMAND ${PYTHON_EXE} -m grpc_tools.protoc -I ${CMAKE_CURRENT_SOURCE_DIR}/protos
        --python_out=${CMAKE_CURRENT_BINARY_DIR} --grpc_python_out ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/protos/metaspore.proto
    DEPENDS testing_venv ${CMAKE_CURRENT_SOURCE_DIR}/protos/metaspore.proto)
add_custom_target(py_grpc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/metaspore_pb2_grpc.py)

if(BUILD_SERVING_BIN)
    add_cpp_test(test_py_preprocessing_process serving/py_preprocessing_process_test.cpp)
    add_custom_command(TARGET test_py_preprocessing_process
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/testing_preprocessor_conf
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/python/scripts/preprocessing/example_requirements.txt
                ${CMAKE_CURRENT_BINARY_DIR}/testing_preprocessor_conf/requirements.txt
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/python/scripts/preprocessing/example_preprocessor.py
                ${CMAKE_CURRENT_BINARY_DIR}/testing_preprocessor_conf/preprocessor.py
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/python/scripts/preprocessing/test_example_preprocessor.py
                ${CMAKE_CURRENT_BINARY_DIR}/testing_preprocessor_conf/test_example_preprocessor.py
        COMMAND ${PYTHON_EXE} -m grpc_tools.protoc
                -I ${CMAKE_CURRENT_SOURCE_DIR}/protos
                --python_out ${CMAKE_CURRENT_BINARY_DIR}/testing_preprocessor_conf
                --grpc_python_out ${CMAKE_CURRENT_BINARY_DIR}/testing_preprocessor_conf
                ${CMAKE_CURRENT_SOURCE_DIR}/protos/metaspore.proto
    )
    add_cpp_test(test_py_preprocessing_model serving/py_preprocessing_model_test.cpp)
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/testing_venv/lib/python3.8/site-packages/metaspore/agent.py
    COMMAND ${PYTHON_EXE} -m pip install --upgrade pip
    COMMAND ${PYTHON_EXE} -m pip install --upgrade --force-reinstall --no-deps ${CMAKE_CURRENT_BINARY_DIR}/${wheel_file_name}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${wheel_file_name} testing_venv
)
add_custom_target(metaspore_wheel_install ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/testing_venv/lib/python3.8/site-packages/metaspore/agent.py python_wheel)

function(add_py_test test_name file_name)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file_name}
        COMMAND ${CMAKE_COMMAND} -E create_hardlink ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/${file_name} ${CMAKE_CURRENT_BINARY_DIR}/${file_name}
        DEPENDS testing_venv py_grpc metaspore_wheel_install)
    add_custom_target(${test_name} ALL DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/${file_name} metaspore_wheel_install copy_files)
    add_test(NAME ${test_name} COMMAND ${PYTHON_EXE} ${CMAKE_CURRENT_BINARY_DIR}/${file_name})
endfunction()

add_py_test(test_dense_xgboost_train dense_xgboost.py)
add_py_test(test_dense_xgboost_grpc dense_xgboost_grpc_test.py)
add_py_test(test_embedding_bag_export.py embedding_bag_export.py)
add_py_test(test_mnist_mlp_train mnist_mlp.py)
add_py_test(test_mnist_mlp_eval mnist_mlp_eval.py)
add_py_test(test_sparse_two_tower_train_export sparse_two_tower_export_demo.py)
add_py_test(test_sparse_mlp_train_export sparse_mlp_export_demo.py)
add_py_test(test_sparse_wdl_train_export sparse_wdl_export_demo.py)
add_py_test(test_sparse_wdl_export sparse_wdl_export_test.py)
add_py_test(test_sparse_wdl_grpc sparse_wdl_grpc_test.py)
add_py_test(test_two_tower_retrieval_milvus two_tower_retrieval_milvus.py)
