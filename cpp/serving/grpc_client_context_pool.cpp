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

#include <serving/grpc_client_context_pool.h>

namespace metaspore::serving {

GrpcClientContextPool::GrpcClientContextPool(size_t thread_count)
    : grpc_client_thread_count_(thread_count)
{
    for (size_t i = 0; i < grpc_client_thread_count_; i++) {
        auto &grpc_context = grpc_client_contexts_.emplace_front(std::make_unique<grpc::CompletionQueue>());
        guards_.emplace_back(grpc_context.get_executor());
    }
    for (size_t i = 0; i < grpc_client_thread_count_; i++) {
        grpc_client_threads_.emplace_back([&, i] {
            auto &grpc_context = *std::next(grpc_client_contexts_.begin(), i);
            grpc_context.run();
        });
    }
}

agrpc::GrpcContext &GrpcClientContextPool::get_next()
{
    return *strategy_.get_next(grpc_client_contexts_.begin(), grpc_client_thread_count_);
}

void GrpcClientContextPool::wait()
{
    guards_.clear();
    for (auto &thread : grpc_client_threads_)
        thread.join();
}

} // namespace metaspore::serving
