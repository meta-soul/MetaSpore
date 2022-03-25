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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <serving/feature_compute_funcs.h>
#include <common/logger.h>
#include <serving/print_utils.h>
#include <serving/threadpool.h>

namespace metaspore::serving {

#define ASSERT_STATUS_OK_COROUTINE(status)                                                         \
    EXPECT_TRUE(status.ok()) << status;                                                            \
    if (!status.ok()) {                                                                            \
        co_return;                                                                                 \
    }

#define ASSERT_TRUE_COROUTINE(expr)                                                                \
    EXPECT_TRUE(expr);                                                                             \
    if (!expr) {                                                                                   \
        co_return;                                                                                 \
    }

#define ASSERT_EQUAL_COROUTINE(lhs, rhs)                                                           \
    EXPECT_EQ(lhs, rhs);                                                                           \
    if (lhs != rhs) {                                                                              \
        co_return;                                                                                 \
    }

int run_all_tests(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    SpdlogDefault::Init();
    auto status = RegisterAllFunctions();
    if (!status.ok()) {
        fmt::print(stderr, "register arrow functions failed {}\n", status);
        return 1;
    }
    testing::InitGoogleTest(&argc, argv);
    auto r = RUN_ALL_TESTS();
    auto &tp = metaspore::serving::Threadpools::get_compute_threadpool();
    auto &btp = metaspore::serving::Threadpools::get_background_threadpool();
    tp.join();
    btp.join();
    tp.stop();
    btp.stop();
    return r;
}

} // namespace metaspore::serving