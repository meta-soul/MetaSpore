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

function(get_python_wheel_name var)
    set(src)
    string(APPEND src "import sys; ")
    string(APPEND src "ver = '%d%d' % ")
    string(APPEND src "(sys.version_info.major, sys.version_info.minor); ")
    string(APPEND src "flag = 'u' if ver == '27' and ")
    string(APPEND src "sys.maxunicode == 0x10ffff else ''; ")
    string(APPEND src "flag += 'm' if ver == '37' else ''; ")
    string(APPEND src "print('cp%s-cp%s%s-linux_x86_64' % ")
    string(APPEND src "(ver, ver, flag), end='')")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "${src}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE wheel_tag)
    if(NOT "${rc}" STREQUAL "0" OR "${wheel_tag}" STREQUAL "")
        message(FATAL_ERROR "Can not get Python wheel tag.")
    endif()
    execute_process(
        COMMAND bash "-c" "grep --color=never version pyproject.toml | grep --color=never -Eo '[0-9\.]+'"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE wheel_version
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${rc}" STREQUAL "0" OR "${wheel_version}" STREQUAL "")
        message(FATAL_ERROR "Can not get Python wheel version.")
    endif()
    set("${var}" "${CMAKE_PROJECT_NAME}-${wheel_version}-${wheel_tag}.whl" PARENT_SCOPE)
endfunction()
