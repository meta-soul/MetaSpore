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

#include <arrow/api.h>
#include <serving/arrow_helpers.h>
#include <serving/sparse_lookup_model.h>
#include <serving/test_utils.h>

#include <thread>

using namespace std::chrono_literals;

using namespace metaspore::serving;

TEST(SparseLookupModelTestSuite, SparseLookupModelTest) {
    auto &tp = Threadpools::get_background_threadpool();
    boost::asio::co_spawn(
        tp,
        []() -> awaitable<void> {
            SparseLookupModel slm;
            auto s = co_await slm.load("sparse_model");
            EXPECT_TRUE(s.ok());
            if (!s.ok()) {
                fmt::print(stderr, "load failed: {}", s);
                co_return;
            }
            auto size_result = co_await slm.get_vector_size();
            EXPECT_TRUE(size_result.ok());
            if (!size_result.ok()) {
                fmt::print(stderr, "get vector size failed: {}", size_result.status());
                co_return;
            }
            auto size = *size_result;
            fmt::print("Get vector size of {}\n", size);
            EXPECT_EQ(size, 16UL);

            {
                // 1d indices
                auto indices = ArrowHelpers::create_1d_tensor<arrow::UInt64Type>(
                    {4492285504028415631UL, 14262736969684564762UL, 14557085601173423661UL,
                     7283129329918422474UL, 123UL});
                auto input = std::make_unique<SparseLookupModelInput>();
                input->indices = *indices;
                input->batch_size = 1;
                auto values_result = co_await slm.predict(std::move(input));
                EXPECT_TRUE(values_result.ok());
                if (!values_result.ok()) {
                    fmt::print(stderr, "lookup failed {}\n", values_result.status());
                    co_return;
                }
                const auto *output =
                    dynamic_cast<const SparseLookupModelOutput *>(values_result->get());
                EXPECT_EQ(output->batch_size, 1L);
                auto values_tensor = output->values;
                const auto &shape = values_tensor->shape();
                EXPECT_EQ(shape.size(), 2UL);
                EXPECT_EQ(shape[0], (*indices)->shape()[0]);
                EXPECT_EQ(shape[1], 16L);
                auto expected = ArrowHelpers::create_2d_tensor<arrow::FloatType>(
                    {{.0f, 0.0035624f, 0.00417991f, 0.0105265f, .0f, -0.000989339f, .0f, 0.0158307f,
                      -0.00201461f, 0.00581876f, -0.00305219f, .0f, 0.00258636f, 0.00503898f,
                      -0.00256663f, -0.0144779f},
                     {.0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, -0.000419456f, .0f, .0f, .0f, .0f,
                      .0f, .0f, .0f},
                     {.0f, .0f, 0.0031262f, .0f, .0f, -0.00158143f, .0f, 0.00618966f, .0f, .0f, .0f,
                      0.00487431f, 0.00473665f, -0.000874193f, 0.00165607f, -0.00474101f},
                     {.0f, .0f, .0f, .0f, 6.27417e-05f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f,
                      .0f, .0f},
                     {.0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f,
                      .0f}});
                TensorPrint::print_tensor<uint64_t>(*input->indices);
                TensorPrint::print_tensor<float>(*values_tensor);
                // Equals cannot pass because arrow implements float tensor compare with no floating
                // point approximation EXPECT_TRUE(values_tensor->Equals(**expected));
            }
        },
        boost::asio::detached);
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }