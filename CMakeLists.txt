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

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(metaspore
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "MetaSpore AI Platform"
    HOMEPAGE_URL "https://github.com/meta-soul/MetaSpore.git"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
endif()

# GCC 11 gives false warning about non-heap object free
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-free-nonheap-object")
endif()

option(BUILD_TRAIN_PKG "Enable training package build" ON)
option(BUILD_SERVING_BIN "Enable serving binary build" ON)
option(ENABLE_GPU "Enable GPU support" ON)
option(ENABLE_TESTS "Enable tests build" OFF)

find_package(Git REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED CONFIG)
find_package(Boost COMPONENTS filesystem system REQUIRED)

include(cmake/metaspore_common.cmake)

if(BUILD_TRAIN_PKG)
    include(cmake/get_project_version.cmake)
    include(cmake/get_python_wheel_tag.cmake)
    include(cmake/metaspore_shared.cmake)
    include(cmake/python_wheel.cmake)
endif()

if(BUILD_SERVING_BIN)
    include(cmake/metaspore_serving.cmake)
endif()

if(ENABLE_TESTS)
    include(cmake/metaspore_tests.cmake)
endif()
