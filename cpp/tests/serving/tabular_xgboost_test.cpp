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

#include <arrow/ipc/json_simple.h>
#include <range/v3/all.hpp>

#include <serving/feature_extraction_model_input.h>
#include <serving/ort_model.h>
#include <serving/tabular_model.h>
#include <serving/test_utils.h>
#include <serving/types.h>

#include <thread>

using namespace ranges;
using namespace metaspore::serving;
using namespace std::string_literals;
using namespace arrow;

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

std::shared_ptr<Array> ArrayFromJSON(const std::shared_ptr<DataType> &type,
                                     util::string_view json) {
    std::shared_ptr<Array> out;
    auto result = ipc::internal::json::ArrayFromJSON(type, json, &out);
    if (!result.ok())
        ::abort();
    return out;
}

std::shared_ptr<RecordBatch> RecordBatchFromJSON(const std::shared_ptr<Schema> &schema,
                                                 util::string_view json) {
    // Parse as a StructArray
    auto struct_type = struct_(schema->fields());
    std::shared_ptr<Array> struct_array = ArrayFromJSON(struct_type, json);

    // Convert StructArray to RecordBatch
    return *RecordBatch::FromStructArray(struct_array);
}

TEST(TabularXGBoostModelTestSuite, TabularXGBoostModelTest) {
    GrpcClientContextPool client_context_pool(1);
    boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [&client_context_pool]() -> awaitable<void> {
            TabularModel model;
            auto status = co_await model.load("xgboost_model", client_context_pool);
            ASSERT_STATUS_OK_COROUTINE(status);

            // construct input record batch
            auto schemas = views::iota(0, 10) | views::transform([](int x) {
                               return arrow::field(fmt::format("field_{}", x), arrow::float32());
                           }) |
                           to<std::vector>();
            auto rb = RecordBatchFromJSON(arrow::schema(std::move(schemas)),
                                          R"([
                    [0.6558618,0.13005558,0.03510657,0.23048967,0.63329154,0.43201634,0.5795548,0.5384891,0.9612295,0.39274803]
                ])");
            fmt::print("Input: {}\n", rb->ToString());
            auto fe_input = std::make_unique<FeatureExtractionModelInput>();
            fe_input->feature_tables["input"] = rb;
            auto result = co_await model.predict(std::move(fe_input));
            ASSERT_STATUS_OK_COROUTINE(result.status());
            auto ort_output = dynamic_cast<OrtModelOutput *>(result->get());
            ASSERT_TRUE_COROUTINE(ort_output);
            for (const auto &[name, value] : ort_output->outputs) {
                auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
                fmt::print("Ort Output Name {}, shape {}\n", name, fmt::join(shape, ","));
                TensorPrint::print_tensor<float>(value);
            }
        },
        boost::asio::detached);
    client_context_pool.wait();
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
