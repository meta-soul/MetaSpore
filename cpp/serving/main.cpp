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
#include <serving/feature_compute_funcs.h>
#include <serving/grpc_server.h>
#include <serving/model_manager.h>
#include <serving/threadpool.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gflags/gflags.h>

namespace metaspore::serving {
DECLARE_string(init_load_path);
DECLARE_uint64(grpc_client_threads);
}

using namespace metaspore::serving;

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto status = RegisterAllFunctions();
    if (!status.ok()) {
        fmt::print(stderr, "register arrow functions failed {}\n", status);
        return 1;
    }

    metaspore::SpdlogDefault::Init();

    int grpc_client_thread_count = FLAGS_grpc_client_threads;
    if (grpc_client_thread_count == 0)
        grpc_client_thread_count = std::thread::hardware_concurrency();
    GrpcClientContextPool client_context_pool(grpc_client_thread_count);

    ModelManager::get_model_manager().init(FLAGS_init_load_path, client_context_pool);

    {
        GrpcServer server(client_context_pool);
        server.run();
    }

    client_context_pool.wait();

    auto &tp = metaspore::serving::Threadpools::get_compute_threadpool();
    auto &btp = metaspore::serving::Threadpools::get_background_threadpool();
    tp.join();
    btp.join();
    tp.stop();
    btp.stop();
    return 0;
}
