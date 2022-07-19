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

find_package(Python REQUIRED COMPONENTS Interpreter Development)
message("Found Python at " ${Python_EXECUTABLE})

function(get_pyarrow_include_dir var)
    set(src)
    string(APPEND src "import pyarrow as pa; ")
    string(APPEND src "print(pa.get_include(), end='')")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "${src}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE include_dir)
    if(NOT "${rc}" STREQUAL "0" OR "${include_dir}" STREQUAL "")
        message(FATAL_ERROR "Can not get pyarrow include dir.")
    endif()
    set("${var}" "${include_dir}" PARENT_SCOPE)
endfunction()

function(get_pyarrow_library_path libname var)
    set(src)
    string(APPEND src "import pyarrow as pa; ")
    string(APPEND src "import glob; ")
    string(APPEND src "print(glob.glob(pa.get_library_dirs()[0] + '/${libname}.so.*')[0], end='')")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "${src}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE libpath)
    if(NOT "${rc}" STREQUAL "0" OR "${libpath}" STREQUAL "")
        message(FATAL_ERROR "Can not get pyarrow ${libname}.so path.")
    endif()
    set("${var}" "${libpath}" PARENT_SCOPE)
endfunction()

function(get_pyarrow_libarrow_path var)
    get_pyarrow_library_path(libarrow libarrow_path)
    set("${var}" "${libarrow_path}" PARENT_SCOPE)
endfunction()

function(get_pyarrow_libarrow_python_path var)
    get_pyarrow_library_path(libarrow_python libarrow_python_path)
    set("${var}" "${libarrow_python_path}" PARENT_SCOPE)
endfunction()

get_pyarrow_include_dir(pyarrow_include_dir)
get_pyarrow_libarrow_path(pyarrow_libarrow_path)
get_pyarrow_libarrow_python_path(pyarrow_libarrow_python_path)

add_library(libarrow SHARED IMPORTED)
set_target_properties(libarrow PROPERTIES
    IMPORTED_LOCATION "${pyarrow_libarrow_path}"
    INTERFACE_INCLUDE_DIRECTORIES "${pyarrow_include_dir}")

add_library(libarrow_python SHARED IMPORTED)
set_target_properties(libarrow_python PROPERTIES
    IMPORTED_LOCATION "${pyarrow_libarrow_python_path}"
    INTERFACE_LINK_LIBRARIES "libarrow")

unset(pyarrow_include_dir)
unset(pyarrow_libarrow_path)
unset(pyarrow_libarrow_python_path)
