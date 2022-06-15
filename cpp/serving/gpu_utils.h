//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>

namespace metaspore::serving {

  class GpuHelper {
    public:
     static bool is_gpu_available() {
       //used to count the device numbers
       int count;
       // get the cuda device count
       cudaGetDeviceCount(&count);
       if (count == 0) {
         spdlog::info("There is no device.");
         return false;
       }
       // find the device >= 1.X
       int i;
       for (i = 0; i < count; ++i) {
         cudaDeviceProp prop;
         if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
           if (prop.major >= 1) {
             printDeviceProp(prop);
             break;
           }
         }
       }
       // if can't find the device
       if (i == count) {
         spdlog::info("There is no device supporting CUDA 1.x.");
         return false;
       }
       return true;
     }
 
     static void printDeviceProp(const cudaDeviceProp& prop) {
       spdlog::info("Device Name : {}.", prop.name);
       spdlog::info("totalGlobalMem : {}.", prop.totalGlobalMem);
       spdlog::info("sharedMemPerBlock : {}.", prop.sharedMemPerBlock);
       spdlog::info("regsPerBlock : {}.", prop.regsPerBlock);
       spdlog::info("warpSize : {}.", prop.warpSize);
       spdlog::info("memPitch : {}.", prop.memPitch);
       spdlog::info("maxThreadsPerBlock : {}.", prop.maxThreadsPerBlock);
       spdlog::info("maxThreadsDim[0 - 2] : {} {} {}.", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
       spdlog::info("maxGridSize[0 - 2] : {} {} {}.", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
       spdlog::info("totalConstMem : {}.", prop.totalConstMem);
       spdlog::info("major.minor : {}.{}.", prop.major, prop.minor);
       spdlog::info("clockRate : {}.", prop.clockRate);
       spdlog::info("textureAlignment : {}.", prop.textureAlignment);
       spdlog::info("deviceOverlap : {}.", prop.deviceOverlap);
       spdlog::info("multiProcessorCount : {}.", prop.multiProcessorCount);
     }
  };
} // namespace metaspore::serving