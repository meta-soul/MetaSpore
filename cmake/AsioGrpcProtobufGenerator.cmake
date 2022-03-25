# From https://github.com/Tradias/asio-grpc/blob/master/cmake/AsioGrpcProtobufGenerator.cmake
# Copyright 2022 Dennis Hezel
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

# Adapted from the original protobuf_generate provided by protobuf-config.cmake
#[=======================================================================[.rst:
asio_grpc_protobuf_generate
------------

Add custom commands to process ``.proto`` files to C++::

asio_grpc_protobuf_generate(PROTOS <proto_file1> [<proto_file2>...]
                            [OUT_DIR <output_directory>]
                            [OUT_VAR <output_variable>]
                            [TARGET <target>]
                            [USAGE_REQUIREMENT PRIVATE|PUBLIC|INTERFACE]
                            [IMPORT_DIRS <directories>...]
                            [EXTRA_ARGS <arguments>...]
                            [GENERATE_GRPC]
                            [GENERATE_DESCRIPTORS])

``PROTOS``
    ``.proto`` files
``OUT_DIR``
    Generated files output directory. Default: CMAKE_CURRENT_BINARY_DIR
``OUT_VAR``
    Variable to define with generated source files
``TARGET``
    Add generated source files to target.
``USAGE_REQUIREMENT``
    How to add sources to ``<target>``: ``PRIVATE``, ``PUBLIC``, ``INTERFACE``
    Default: ``PRIVATE``
``IMPORT_DIRS``
    Additional import directories to be added to the protoc command line
``EXTRA_ARGS``
    Additional protoc command line arguments
``GENERATE_GRPC``
    Generate grpc files
``GENERATE_DESCRIPTORS``
    Generate descriptor files named <proto_file_base_name>.desc

#]=======================================================================]
# // begin-snippet: asio_grpc_protobuf_generate
function(asio_grpc_protobuf_generate)
    # // end-snippet: asio_grpc_protobuf_generate
    include(CMakeParseArguments)

    set(_asio_grpc_options GENERATE_GRPC GENERATE_DESCRIPTORS)
    set(_asio_grpc_singleargs OUT_VAR OUT_DIR TARGET USAGE_REQUIREMENT)
    set(_asio_grpc_multiargs PROTOS IMPORT_DIRS EXTRA_ARGS)

    cmake_parse_arguments(asio_grpc_protobuf_generate "${_asio_grpc_options}" "${_asio_grpc_singleargs}"
                          "${_asio_grpc_multiargs}" "${ARGN}")

    if(NOT asio_grpc_protobuf_generate_PROTOS)
        message(SEND_ERROR "asio_grpc_protobuf_generate called without any proto files: PROTOS")
        return()
    endif()

    if(NOT asio_grpc_protobuf_generate_OUT_VAR AND NOT asio_grpc_protobuf_generate_TARGET)
        message(SEND_ERROR "asio_grpc_protobuf_generate called without a target or output variable: TARGET or OUT_VAR")
        return()
    endif()

    if(NOT asio_grpc_protobuf_generate_OUT_DIR)
        set(asio_grpc_protobuf_generate_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    set(_asio_grpc_generated_extensions ".pb.cc" ".pb.h")
    if(asio_grpc_protobuf_generate_GENERATE_GRPC)
        list(APPEND _asio_grpc_generated_extensions ".grpc.pb.cc" ".grpc.pb.h")
    endif()

    if(NOT asio_grpc_protobuf_generate_USAGE_REQUIREMENT)
        set(asio_grpc_protobuf_generate_USAGE_REQUIREMENT "PRIVATE")
    endif()

    # Create an include path for each file specified
    foreach(_asio_grpc_file ${asio_grpc_protobuf_generate_PROTOS})
        get_filename_component(_asio_grpc_abs_file "${_asio_grpc_file}" ABSOLUTE)
        get_filename_component(_asio_grpc_abs_path "${_asio_grpc_abs_file}" PATH)
        list(FIND _asio_grpc_protobuf_include_path "${_asio_grpc_abs_path}" _asio_grpc_contains_already)
        if(${_asio_grpc_contains_already} EQUAL -1)
            list(APPEND _asio_grpc_protobuf_include_path -I "${_asio_grpc_abs_path}")
        endif()
    endforeach()

    # Add all explicitly specified import directory to the include path
    foreach(_asio_grpc_dir ${asio_grpc_protobuf_generate_IMPORT_DIRS})
        get_filename_component(_asio_grpc_abs_dir "${_asio_grpc_dir}" ABSOLUTE)
        list(FIND _asio_grpc_protobuf_include_path "${_asio_grpc_abs_dir}" _asio_grpc_contains_already)
        if(${_asio_grpc_contains_already} EQUAL -1)
            list(APPEND _asio_grpc_protobuf_include_path -I "${_asio_grpc_abs_dir}")
        endif()
    endforeach()

    set(_asio_grpc_generated_srcs_all)
    foreach(_asio_grpc_proto ${asio_grpc_protobuf_generate_PROTOS})
        get_filename_component(_asio_grpc_abs_file "${_asio_grpc_proto}" ABSOLUTE)
        get_filename_component(_asio_grpc_basename "${_asio_grpc_proto}" NAME_WE)

        # Collect generated files
        set(_asio_grpc_generated_srcs)
        foreach(_asio_grpc_ext ${_asio_grpc_generated_extensions})
            list(APPEND _asio_grpc_generated_srcs
                 "${asio_grpc_protobuf_generate_OUT_DIR}/${_asio_grpc_basename}${_asio_grpc_ext}")
        endforeach()

        if(asio_grpc_protobuf_generate_GENERATE_DESCRIPTORS)
            set(_asio_grpc_descriptor_file "${asio_grpc_protobuf_generate_OUT_DIR}/${_asio_grpc_basename}.desc")
            set(_asio_grpc_descriptor_command "--descriptor_set_out=${_asio_grpc_descriptor_file}")
            list(APPEND _asio_grpc_generated_srcs "${_asio_grpc_descriptor_file}")
        endif()
        list(APPEND _asio_grpc_generated_srcs_all ${_asio_grpc_generated_srcs})

        # Run protoc
        set(_asio_grpc_command_arguments --cpp_out "${asio_grpc_protobuf_generate_OUT_DIR}"
                                         "${_asio_grpc_descriptor_command}" "${_asio_grpc_protobuf_include_path}")
        if(asio_grpc_protobuf_generate_GENERATE_GRPC)
            list(APPEND _asio_grpc_command_arguments --grpc_out "${asio_grpc_protobuf_generate_OUT_DIR}"
                 "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>")
        endif()
        list(APPEND _asio_grpc_command_arguments ${asio_grpc_protobuf_generate_EXTRA_ARGS} "${_asio_grpc_abs_file}")
        string(REPLACE ";" " " _asio_grpc_pretty_command_arguments "${_asio_grpc_command_arguments}")
        add_custom_command(
            OUTPUT ${_asio_grpc_generated_srcs}
            COMMAND "${CMAKE_COMMAND}" "-E" "make_directory" "${asio_grpc_protobuf_generate_OUT_DIR}"
            COMMAND protobuf::protoc ${_asio_grpc_command_arguments}
            MAIN_DEPENDENCY "${_asio_grpc_abs_file}"
            DEPENDS protobuf::protoc
            COMMENT "protoc ${_asio_grpc_pretty_command_arguments}"
            VERBATIM)
    endforeach()

    set_source_files_properties(${_asio_grpc_generated_srcs_all} PROPERTIES SKIP_UNITY_BUILD_INCLUSION on)

    if(asio_grpc_protobuf_generate_TARGET)
        target_sources(${asio_grpc_protobuf_generate_TARGET} "${asio_grpc_protobuf_generate_USAGE_REQUIREMENT}"
                                                             ${_asio_grpc_generated_srcs_all})
    endif()

    if(asio_grpc_protobuf_generate_OUT_VAR)
        set(${asio_grpc_protobuf_generate_OUT_VAR}
            ${_asio_grpc_generated_srcs_all}
            PARENT_SCOPE)
    endif()
endfunction()
