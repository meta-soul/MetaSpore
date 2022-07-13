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

#include <fmt/ranges.h>
#include <serving/arrow_helpers.h>
#include <serving/feature_extraction_model_input.h>
#include <serving/ort_model.h>
#include <serving/tabular_model.h>
#include <serving/test_utils.h>
#include <serving/threadpool.h>
#include <common/types.h>

#include <boost/asio/use_future.hpp>

using namespace metaspore::serving;
using namespace std::string_literals;

static std::pair<std::shared_ptr<arrow::RecordBatch>, std::shared_ptr<arrow::Schema>>
make_user_record_batch() {
    auto user_id =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"user_id_0", "user_id_1"})
            .ValueOrDie();
    auto user_age =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"20", "25"}).ValueOrDie();
    auto user_sex =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"male", "female"}).ValueOrDie();
    auto user_city =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"beijing", "shanghai"})
            .ValueOrDie();
    auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
        arrow::field("user_id", arrow::utf8()), arrow::field("user_age", arrow::utf8()),
        arrow::field("user_sex", arrow::utf8()), arrow::field("user_city", arrow::utf8())});
    auto batch = ArrowHelpers::GetSampleRecordBatch({user_id, user_age, user_sex, user_city},
                                                    schema->fields())
                     .ValueOrDie();
    return std::make_pair(batch, schema);
}

static std::pair<std::shared_ptr<arrow::RecordBatch>, std::shared_ptr<arrow::Schema>>
make_item_record_batch() {
    auto user_idarray =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"user_id_0", "user_id_1"})
            .ValueOrDie();
    auto item_category =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"clothing", "shoes"})
            .ValueOrDie();
    auto item_color =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"white", "black"}).ValueOrDie();
    auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
        arrow::field("user_id", arrow::utf8()), arrow::field("item_category", arrow::utf8()),
        arrow::field("item_color", arrow::utf8())});
    auto batch = ArrowHelpers::GetSampleRecordBatch({user_idarray, item_category, item_color},
                                                    schema->fields())
                     .ValueOrDie();
    return std::make_pair(batch, schema);
}

TEST(TabularModelTestSuite, TabularModelTest) {
    auto f = boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        []() -> awaitable<void> {
            TabularModel model;
            auto status = co_await model.load("sparse_ctr_model");
            ASSERT_STATUS_OK_COROUTINE(status);
            auto [user_batch, user_schema] = make_user_record_batch();
            auto [item_batch, item_schema] = make_item_record_batch();

            auto fe_input = std::make_unique<FeatureExtractionModelInput>();
            fe_input->feature_tables["user_table"] = user_batch;
            fe_input->feature_tables["item_table"] = item_batch;
            auto result = co_await model.do_predict(std::move(fe_input));
            status = result.status();
            ASSERT_STATUS_OK_COROUTINE(status);
            ASSERT_EQUAL_COROUTINE((*result)->outputs.size(), 1UL);
            for (const auto &[name, value] : (*result)->outputs) {
                fmt::print("TabularModel produced \"{}\", ort type {}\n", name,
                           value.GetTypeInfo().GetONNXType());
                ASSERT_TRUE_COROUTINE(value.IsTensor());
                auto tsi = value.GetTensorTypeAndShapeInfo();
                fmt::print("Dims {}, shape [{}]\n", tsi.GetDimensionsCount(),
                           fmt::join(tsi.GetShape(), ", "));
                ASSERT_EQUAL_COROUTINE(tsi.GetDimensionsCount(), 2UL);
                auto shape = tsi.GetShape();
                ASSERT_EQUAL_COROUTINE(shape[0], 2);
                ASSERT_EQUAL_COROUTINE(shape[1], 1);
                TensorPrint::print_tensor<float>(value);
            }
        },
        boost::asio::use_future);
    (void)f.get();
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
