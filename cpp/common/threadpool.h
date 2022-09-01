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

#include <stdlib.h>
#include <boost/asio/thread_pool.hpp>
#include <gflags/gflags.h>

namespace metaspore {

DECLARE_uint64(compute_thread_num);
DECLARE_uint64(background_thread_num);

using threadpool = boost::asio::thread_pool;

class Threadpools {
  public:
    static threadpool &get_compute_threadpool() {
        static threadpool tp(get_compute_thread_num());
        return tp;
    }

    static threadpool &get_background_threadpool() {
        static threadpool tp(get_background_thread_num());
        return tp;
    }

  private:
    static int get_compute_thread_num() {
        const char* str = getenv("METASPORE_COMPUTE_THREAD_NUM");
        if (str)
            return std::stoi(str);
        return FLAGS_compute_thread_num;
    }

    static int get_background_thread_num() {
        const char* str = getenv("METASPORE_BACKGROUND_THREAD_NUM");
        if (str)
            return std::stoi(str);
        return FLAGS_background_thread_num;
    }
};

} // namespace metaspore
