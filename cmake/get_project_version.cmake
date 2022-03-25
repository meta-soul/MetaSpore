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

function(get_project_version var)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE commit_id)
    if(NOT "${rc}" STREQUAL "0" OR "${commit_id}" STREQUAL "")
        message(FATAL_ERROR "Can not find commit id of the repository.")
    endif()
    string(STRIP "${commit_id}" commit_id)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --short
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out)
    if(NOT "${rc}" STREQUAL "0")
        message(FATAL_ERROR "Can not check cleanness the repository.")
    endif()
    string(STRIP "${out}" out)
    if(NOT "${out}" STREQUAL "")
        set(commit_id "${commit_id}.dirty")
    endif()
    set(version)
    string(APPEND version ${PROJECT_VERSION_MAJOR}.)
    string(APPEND version ${PROJECT_VERSION_MINOR}.)
    string(APPEND version ${PROJECT_VERSION_PATCH}+)
    string(APPEND version ${commit_id})
    set("${var}" "${version}" PARENT_SCOPE)
endfunction()
