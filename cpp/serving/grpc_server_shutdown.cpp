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

#include <grpcpp/server.h>
#include <common/logger.h>
#include <serving/grpc_server_shutdown.h>

namespace metaspore::serving {

GrpcServerShutdown::GrpcServerShutdown(grpc::Server &server, agrpc::GrpcContext &grpc_context)
    : server_(server), signals_(grpc_context, SIGINT, SIGTERM) {
    signals_.async_wait([&](auto &&ec, auto &&signal) {
        if (boost::asio::error::operation_aborted != ec) {
            spdlog::info("Shutdown with signal {}", signal);
            shutdown();
        }
    });
}

GrpcServerShutdown::~GrpcServerShutdown() {
    if (shutdown_thread_.joinable()) {
        shutdown_thread_.join();
    } else if (!is_shutdown_.exchange(true)) {
        server_.Shutdown();
    }
}

void GrpcServerShutdown::shutdown() {
    if (!is_shutdown_.exchange(true)) {
        // We cannot call server.Shutdown() on the same thread that runs a GrpcContext
        // because that could lead to deadlock, therefore create a new thread.
        shutdown_thread_ = std::thread([&] {
            signals_.cancel();
            server_.Shutdown();
        });
    }
}

} // namespace metaspore::serving
