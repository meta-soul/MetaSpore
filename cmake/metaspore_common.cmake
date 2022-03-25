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

find_package(gflags REQUIRED)

set(SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/globals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/hashtable_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/hash_uniquifier.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/map_file_header.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/memory_usage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/hashmap/memory_mapped_array_hash_map_loader.cpp
)

add_library(metaspore-common STATIC
    ${SRCS}
)

target_include_directories(metaspore-common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

target_link_libraries(metaspore-common PUBLIC
    gflags::gflags
)