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

#include <cuda_runtime.h>

namespace metaspore::serving {

class GpuHelper {
  public:
    static bool is_gpu_available() {
        // used to count the device numbers
        int deviceCount = 0;

        // get the cuda device count
        cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
        spdlog::info("deviceCount: {}", deviceCount);

        if (deviceCount <= 0) {
            spdlog::info("No device detected");
            return false;
        }

        // Error when running cudaGetDeviceCount
        if (cudaResultCode != cudaSuccess) {
            spdlog::info("{} (CUDA error Code={})", cudaGetErrorString(cudaResultCode),
                         (int)cudaResultCode);
            return false;
        }
        return true;
    }
};
} // namespace metaspore::serving