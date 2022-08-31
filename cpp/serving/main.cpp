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

#include <common/logger.h>
#include <common/features/feature_compute_funcs.h>
#include <serving/grpc_server.h>
#include <serving/grpc_client_context_pool.h>
#include <serving/model_manager.h>
#include <common/threadpool.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gflags/gflags.h>

namespace metaspore::serving {
DECLARE_string(init_load_path);
}

using namespace metaspore::serving;

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto status = metaspore::RegisterCustomArrowFunctions();
    if (!status.ok()) {
        fmt::print(stderr, "register arrow functions failed {}\n", status);
        return 1;
    }

    metaspore::SpdlogDefault::Init();
    GrpcClientContextPool::get_instance();

    ModelManager::get_model_manager().init(FLAGS_init_load_path);

    {
        GrpcServer server;
        server.run();
    }

    GrpcClientContextPool::get_instance().wait();
    auto &tp = metaspore::Threadpools::get_compute_threadpool();
    auto &btp = metaspore::Threadpools::get_background_threadpool();
    tp.join();
    btp.join();
    tp.stop();
    btp.stop();
    return 0;
}
