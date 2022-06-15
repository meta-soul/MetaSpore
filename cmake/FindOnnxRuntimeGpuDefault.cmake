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

find_path(ORT_GPU_INCLUDE_DIR onnxruntime_cxx_api.h )
find_library(ORT_GPU_LIBRARY onnxruntime)

add_library( onnxruntime-gpu-default SHARED IMPORTED )
set_property(TARGET onnxruntime-gpu-default PROPERTY
             IMPORTED_LOCATION ${ORT_GPU_LIBRARY})
target_include_directories(onnxruntime-gpu-default INTERFACE ${ORT_GPU_INCLUDE_DIR})