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

#include <serving/arrow_helpers.h>
#include <serving/feature_compute_exec.h>
#include <serving/test_utils.h>
#include <common/types.h>
#include <serving/utils.h>

#include <arrow/compute/api.h>
#include <boost/asio/use_future.hpp>

using namespace metaspore::serving;
using namespace std::string_literals;
namespace cp = arrow::compute;

static std::pair<std::shared_ptr<arrow::RecordBatch>, std::shared_ptr<arrow::Schema>>
make_user_record_batch() {
    auto idarray =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"user_id_0"}).ValueOrDie();
    auto user_feature =
        ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>({"user_feature_0"}).ValueOrDie();
    auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
        arrow::field("user_id", arrow::utf8()), arrow::field("user_feature", arrow::utf8())});
    auto batch =
        ArrowHelpers::GetSampleRecordBatch({idarray, user_feature}, schema->fields()).ValueOrDie();
    return std::make_pair(batch, schema);
}

static std::pair<std::shared_ptr<arrow::RecordBatch>, std::shared_ptr<arrow::Schema>>
make_item_record_batch() {
    auto user_idarray = ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>(
                            {"user_id_0", "user_id_0", "user_id_1"})
                            .ValueOrDie();
    auto campaign_id = ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>(
                           {"campaign_id_0", "campaign_id_1", "campaign_id_2"})
                           .ValueOrDie();
    auto campaign_feature = ArrowHelpers::GetBinaryArrayDataSample<arrow::StringType>(
                                {"campaign_fea_0", "campaign_fea_1", "campaign_fea_2"})
                                .ValueOrDie();
    auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
        arrow::field("user_id", arrow::utf8()), arrow::field("campaign_id", arrow::utf8()),
        arrow::field("campaign_feature", arrow::utf8())});
    auto batch = ArrowHelpers::GetSampleRecordBatch({user_idarray, campaign_id, campaign_feature},
                                                    schema->fields())
                     .ValueOrDie();
    return std::make_pair(batch, schema);
}

TEST(FeatureComputeExecTestSuite, FeatureComputeJoinTest) {
    auto fn = [&]() -> boost::asio::awaitable<void> {
        FeatureComputeExec exec;
        std::string user_table("user_table");
        std::string item_table("item_table");
        auto [user_batch, user_schema] = make_user_record_batch();
        auto status = exec.add_source(user_table);
        ASSERT_STATUS_OK_COROUTINE(status);
        auto [item_batch, item_schema] = make_item_record_batch();
        status = exec.add_source(item_table);
        ASSERT_STATUS_OK_COROUTINE(status);

        status = exec.add_join_plan(item_table, user_table, arrow::compute::JoinType::LEFT_OUTER,
                                    std::vector({"user_id"s}), std::vector({"user_id"s}));
        ASSERT_STATUS_OK_COROUTINE(status);

        status = exec.add_projection(
            {cp::call("bkdr_hash", {cp::field_ref("campaign_id")},
                      StringBKDRHashFunctionOption::Make("campaign_id")),
             cp::call("bkdr_hash_combine",
                      {cp::call("bkdr_hash", {cp::field_ref("user_feature")},
                                StringBKDRHashFunctionOption::Make("user_feature")),
                       cp::call("bkdr_hash", {cp::field_ref("campaign_feature")},
                                StringBKDRHashFunctionOption::Make("campaign_feature"))},
                      BKDRHashCombineFunctionOption::Make())});

        auto ctx = exec.start_plan();
        status = ctx.status();
        ASSERT_STATUS_OK_COROUTINE(status);

        status = exec.set_input_schema(*ctx, user_table, user_schema);
        ASSERT_STATUS_OK_COROUTINE(status);
        status = exec.set_input_schema(*ctx, item_table, item_schema);
        ASSERT_STATUS_OK_COROUTINE(status);

        status = exec.build_plan(*ctx);
        ASSERT_STATUS_OK_COROUTINE(status);
        // feed left table, item
        status = exec.feed_input(*ctx, item_table, item_batch);
        ASSERT_STATUS_OK_COROUTINE(status);
        // feed right table, user
        status = exec.feed_input(*ctx, user_table, user_batch);
        ASSERT_STATUS_OK_COROUTINE(status);
        // get output record batch
        auto output_result = co_await exec.execute(*ctx);
        status = output_result.status();
        ASSERT_STATUS_OK_COROUTINE(status);
        auto record_batch = *output_result;
        fmt::print("output batch1: {}\n", record_batch->ToString());
        status = exec.finish_plan(*ctx);
        ASSERT_STATUS_OK_COROUTINE(status);
        co_return;
    };

    auto fut = boost::asio::co_spawn(Threadpools::get_background_threadpool(), std::move(fn),
                                     boost::asio::use_future);
    fut.get();
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
