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

#include <iostream>

#include <gflags/gflags.h>

#ifndef SPDLOG_ACTIVE_LEVEL
#ifndef NDEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif
#endif

#include <spdlog/spdlog.h>

#include <chrono>

namespace metaspore {

DECLARE_int32(log_level);

class SpdlogDefault {
  public:
    SpdlogDefault() {
        spdlog::set_pattern("[%m%d-%H:%M:%S:%e][%l][T%t] %v");
        spdlog::set_level((spdlog::level::level_enum)FLAGS_log_level);
        spdlog::flush_every(std::chrono::seconds(5));
        std::cerr << "Logger macro level " << (SPDLOG_ACTIVE_LEVEL)
                  << ", runtime level: " << FLAGS_log_level << std::endl;
    }

    ~SpdlogDefault() { spdlog::shutdown(); }

    static void Init() { static SpdlogDefault logger; }
};

} // namespace metaspore