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

#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/process/system.hpp>
#include <boost/process/search_path.hpp>
#include <boost/asio/use_future.hpp>
#include <serving/test_utils.h>
#include <serving/py_preprocessing_model.h>

using namespace metaspore;
using namespace metaspore::serving;

#define EXPECT_TRUE_COROUTINE(status)                         \
    EXPECT_TRUE(status.ok()) << status;                       \
    if (!status.ok()) {                                       \
        co_return;                                            \
    }

TEST(PyPreprocessingModelTestSuite, LoadModelTest) {
    auto &tp = Threadpools::get_background_threadpool();
    std::future<void> future = boost::asio::co_spawn(
        tp,
        []() -> awaitable<void> {
            auto prog_dir = boost::dll::program_location().parent_path();
            auto conf_dir = prog_dir / "testing_preprocessor_conf";
            PyPreprocessingModel model;
            auto status = co_await model.load(conf_dir.string());
            EXPECT_TRUE_COROUTINE(status);
        },
        boost::asio::use_future);
    future.get();
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
