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

#include <atomic>
#include <thread>
#include <agrpc/asioGrpc.hpp>
#include <boost/asio/signal_set.hpp>

namespace metaspore::serving {

class GrpcServerShutdown {
public:
    GrpcServerShutdown(grpc::Server &server, agrpc::GrpcContext &grpc_context);
    ~GrpcServerShutdown();

    void shutdown();

private:
    grpc::Server &server_;
    boost::asio::basic_signal_set<agrpc::GrpcContext::executor_type> signals_;
    std::atomic_bool is_shutdown_{};
    std::thread shutdown_thread_;
};

} // namespace metaspore::serving
