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

#include <stdint.h>
#include <atomic>
#include <vector>
#include <forward_list>
#include <thread>
#include <agrpc/asioGrpc.hpp>

namespace metaspore::serving {

class GrpcClientContextPool {
public:
    GrpcClientContextPool(size_t thread_count);
    agrpc::GrpcContext &get_next();
    void wait();

private:
    class RoundRobin
    {
    public:
        template<typename Iterator>
        Iterator get_next(Iterator begin, size_t size)
        {
            const size_t cur = current_.fetch_add(1, std::memory_order_relaxed);
            const size_t pos = cur % size;
            return std::next(begin, pos);
        }

    private:
        std::atomic_size_t current_{};
    };

    std::forward_list<agrpc::GrpcContext> grpc_client_contexts_;
    std::vector<boost::asio::executor_work_guard<agrpc::GrpcContext::executor_type>> guards_;
    std::vector<std::thread> grpc_client_threads_;
    size_t grpc_client_thread_count_;
    RoundRobin strategy_;
};

} // namespace metaspore::serving
