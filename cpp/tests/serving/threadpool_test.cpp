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

#include <common/types.h>
#include <serving/test_utils.h>

#include <iostream>
#include <thread>

#include <boost/asio/use_future.hpp>

using namespace metaspore;
using namespace metaspore::serving;

TEST(THREADPOOL_TEST_SUITE, TestThreadPoolBasicSpawn) {
    auto &tp = Threadpools::get_compute_threadpool();
    auto &btp = Threadpools::get_background_threadpool();
    auto f = boost::asio::co_spawn(
        tp,
        [&]() -> awaitable<int> {
            std::cout << "inside compute thread " << std::this_thread::get_id() << std::endl
                      << std::flush;
            co_await boost::asio::co_spawn(
                btp,
                [&]() -> awaitable<void> {
                    std::cout << "inside bg thread " << std::this_thread::get_id() << std::endl
                              << std::flush;
                    co_return;
                },
                boost::asio::use_awaitable);
            std::cout << "after bg task finished " << std::this_thread::get_id() << std::endl
                      << std::flush;
            co_return 0;
        },
        boost::asio::use_future);
    (void)f.get();
}

TEST(THREADPOOL_TEST_SUITE, TestThreadPoolNestedSpawn) {
    auto &tp = Threadpools::get_compute_threadpool();
    auto &btp = Threadpools::get_background_threadpool();
    auto f = boost::asio::co_spawn(
        tp,
        [&]() -> awaitable<int> {
            std::cout << "inside compute thread 0: " << std::this_thread::get_id() << std::endl
                      << std::flush;
            co_await boost::asio::co_spawn(
                tp,
                [&]() -> awaitable<void> {
                    std::cout << "inside compute thread 1: " << std::this_thread::get_id()
                              << std::endl
                              << std::flush;
                    co_await boost::asio::co_spawn(
                        tp,
                        [&]() -> awaitable<void> {
                            std::cout << "inside compute thread 2: " << std::this_thread::get_id()
                                      << std::endl
                                      << std::flush;
                            co_return;
                        },
                        boost::asio::use_awaitable);
                    std::cout << "return to coroutine 1: " << std::this_thread::get_id()
                              << std::endl
                              << std::flush;
                    co_return;
                },
                boost::asio::use_awaitable);
            std::cout << "return to coroutine 0: " << std::this_thread::get_id() << std::endl
                      << std::flush;
            co_return 0;
        },
        boost::asio::use_future);
    (void)f.get();
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
