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
#include <arrow/compute/api.h>
#include <arrow/compute/exec/exec_plan.h>
#include <arrow/compute/exec/options.h>
#include <arrow/util/async_generator.h>

#include <absl/status/status.h>
#include <boost/asio/experimental/concurrent_channel.hpp>
#include <fmt/format.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>
#include <spdlog/spdlog.h>

#include <common/arrow/arrow_helpers.h>
#include <common/threadpool.h>
#include <common/utils.h>

using namespace metaspore;
using namespace std::string_literals;

using channel_type = boost::asio::experimental::concurrent_channel<void(boost::system::error_code,
                                                                        arrow::compute::ExecBatch)>;

class FeatureComputeContext {
  public:
    struct InputSource {
        arrow::PushGenerator<arrow::util::optional<arrow::compute::ExecBatch>> input_queue;
        arrow::compute::ExecNode *node;
    };

    std::shared_ptr<arrow::compute::ExecPlan> plan_;
    std::unordered_map<std::string, InputSource> name_source_map_;
    arrow::compute::ExecNode *root_node_{nullptr};
    arrow::compute::ExecNode *join_node_{nullptr};
    channel_type channel_{Threadpools::get_compute_threadpool(), 10};
    arrow::Future<> sink_future_{arrow::Future<>::Make()};
};

absl::Status add_source(std::unique_ptr<FeatureComputeContext> &context_, const std::string &name,
                        std::shared_ptr<arrow::Schema> schema) {
    auto pair = context_->name_source_map_.emplace(name, FeatureComputeContext::InputSource());
    if (!pair.second) {
        return absl::AlreadyExistsError(fmt::format("Input source {} already exists", name));
    }
    auto node_result = arrow::compute::MakeExecNode(
        "source", context_->plan_.get(), /* inputs = */ {},
        arrow::compute::SourceNodeOptions{schema, pair.first->second.input_queue});
    if (!node_result.ok()) {
        return absl::InternalError(node_result.status().message());
    }
    pair.first->second.node = *node_result;
    context_->root_node_ = *node_result;
    return absl::OkStatus();
}

absl::Status feed_input(std::unique_ptr<FeatureComputeContext> &context_,
                        const std::string &source_name, std::shared_ptr<arrow::RecordBatch> batch) {
    auto source = context_->name_source_map_.find(source_name);
    if (source == context_->name_source_map_.end()) {
        return absl::NotFoundError(
            fmt::format("FeatureComputeExec feed_input with non-exist name {}", source_name));
    }
    arrow::compute::ExecBatch exec_batch(*batch);
    source->second.input_queue.producer().Push(
        arrow::util::make_optional<arrow::compute::ExecBatch>(std::move(exec_batch)));
    return absl::OkStatus();
}

absl::Status add_join_plan(std::unique_ptr<FeatureComputeContext> &context_,
                           const std::string &left_source_name,
                           const std::string &right_source_name, arrow::compute::JoinType join_type,
                           const std::vector<std::string> &left_key_names,
                           const std::vector<std::string> &right_key_names) {
    auto left_source = context_->name_source_map_.find(left_source_name);
    if (left_source == context_->name_source_map_.end()) {
        return absl::NotFoundError(
            fmt::format("FeatureComputeExec feed_input with non-exist name {}", left_source_name));
    }
    auto right_source = context_->name_source_map_.find(right_source_name);
    if (right_source == context_->name_source_map_.end()) {
        return absl::NotFoundError(
            fmt::format("FeatureComputeExec feed_input with non-exist name {}", right_source_name));
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
    arrow::compute::HashJoinNodeOptions join_opts{
        join_type, left_key_names | ranges::views::transform([](const std::string &name) {
                       return arrow::FieldRef(name);
                   }) | ranges::to<std::vector>(),
        right_key_names | ranges::views::transform([](const std::string &name) {
            return arrow::FieldRef(name);
        }) | ranges::to<std::vector>()};
#pragma GCC diagnostic pop

    auto hashjoin = arrow::compute::MakeExecNode(
        "hashjoin", context_->plan_.get(), {left_source->second.node, right_source->second.node},
        join_opts);
    if (!hashjoin.ok()) {
        return absl::InternalError(hashjoin.status().message());
    }
    context_->root_node_ = *hashjoin;
    context_->join_node_ = *hashjoin;
    return absl::OkStatus();
}

struct SinkConsumer : public arrow::compute::SinkNodeConsumer {

    SinkConsumer(std::shared_ptr<arrow::Schema> schema, channel_type &ch, arrow::Future<> fut)
        : output_schema_(schema), channel_(ch), fut_(std::move(fut)) {}

    arrow::Status Consume(arrow::compute::ExecBatch batch) override {
        fmt::print("consuming batch\n");
        bool s = channel_.try_send(boost::system::error_code(), std::move(batch));
        if (!s) {
            fmt::print("send batch failed\n");
            return arrow::Status::CapacityError("cannot consume batch and send to channel");
        } else {
            fmt::print("send batch succeeded\n");
            return arrow::Status::OK();
        }
    }

    arrow::Future<> Finish() override {
        fmt::print("consumer finished\n");
        return fut_;
    }

    std::shared_ptr<arrow::Schema> output_schema_;
    channel_type &channel_;
    arrow::Future<> fut_;
};

absl::Status finish_plan(std::unique_ptr<FeatureComputeContext> &context_) {
    // create sink reader
    auto sink_result = arrow::compute::MakeExecNode(
        "consuming_sink", context_->plan_.get(), {context_->root_node_},
        arrow::compute::ConsumingSinkNodeOptions{std::make_shared<SinkConsumer>(
            context_->root_node_->output_schema(), context_->channel_, context_->sink_future_)});
    if (!sink_result.ok()) {
        return absl::InternalError(sink_result.status().message());
    }

    // validate the ExecPlan
    auto validate = context_->plan_->Validate();
    if (!validate.ok()) {
        return absl::InternalError(validate.message());
    }
    spdlog::info("FeatureComputeExec created plan {}", context_->plan_->ToString());
    // start the ExecPlan
    auto start = context_->plan_->StartProducing();
    if (!start.ok()) {
        return absl::InternalError(start.message());
    }
    return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<arrow::RecordBatch>>
get_output(std::unique_ptr<FeatureComputeContext> &context_) {
    arrow::compute::ExecBatch exec_batch;
    std::mutex m;
    std::condition_variable cv;
    bool ready = false;
    context_->channel_.async_receive(
        [&](boost::system::error_code ec, arrow::compute::ExecBatch b) {
            fmt::print("exec batch received\n");
            exec_batch = std::move(b);
            {
                std::lock_guard<std::mutex> lk(m);
                ready = true;
            }
            cv.notify_one();
        });
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return ready; });
    auto rb_result = exec_batch.ToRecordBatch(context_->root_node_->output_schema());
    if (!rb_result.ok()) {
        return absl::InternalError(rb_result.status().message());
    }
    return *rb_result;
}

absl::Status stop(std::unique_ptr<FeatureComputeContext> &context_) {
    for (auto &[name, queue] : context_->name_source_map_) {
        queue.input_queue.producer().Close();
    }
    context_->sink_future_.MarkFinished();
    const auto &s = context_->plan_->finished().status();
    if (!s.ok()) {
        return absl::InternalError(fmt::format("Finish plan failed {}", s.message()));
    }
    return absl::OkStatus();
}

static std::pair<std::shared_ptr<arrow::RecordBatch>, std::shared_ptr<arrow::Schema>>
make_user_record_batch() {
    auto idarray = ArrowHelpers::GetArrayDataSample<arrow::UInt64Type>({123UL}).ValueOrDie();
    auto user_feature = ArrowHelpers::GetArrayDataSample<arrow::UInt64Type>({456UL}).ValueOrDie();
    auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
        arrow::field("user_id", arrow::uint64()), arrow::field("user_feature", arrow::uint64())});
    auto batch =
        ArrowHelpers::GetSampleRecordBatch({idarray, user_feature}, schema->fields()).ValueOrDie();
    return std::make_pair(batch, schema);
}

static std::pair<std::shared_ptr<arrow::RecordBatch>, std::shared_ptr<arrow::Schema>>
make_item_record_batch() {
    auto user_idarray =
        ArrowHelpers::GetArrayDataSample<arrow::UInt64Type>({123UL, 123UL, 0UL}).ValueOrDie();
    auto campaign_id =
        ArrowHelpers::GetArrayDataSample<arrow::UInt64Type>({0UL, 1UL, 2UL}).ValueOrDie();
    auto campaign_feature =
        ArrowHelpers::GetArrayDataSample<arrow::UInt64Type>({789UL, 1024UL, 42UL}).ValueOrDie();
    auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
        arrow::field("user_id", arrow::uint64()), arrow::field("campaign_id", arrow::uint64()),
        arrow::field("campaign_feature", arrow::uint64())});
    auto batch = ArrowHelpers::GetSampleRecordBatch({user_idarray, campaign_id, campaign_feature},
                                                    schema->fields())
                     .ValueOrDie();
    return std::make_pair(batch, schema);
}

int main(int argc, char **argv) {

    auto result = arrow::compute::ExecPlan::Make();
    if (!result.ok()) {
        fmt::print(stderr, "{}\n", result.status());
        return 1;
    }
    std::unique_ptr<FeatureComputeContext> context_ = std::make_unique<FeatureComputeContext>();
    context_->plan_ = *result;
    std::string user_table("user_table");
    std::string item_table("item_table");
    auto [user_batch, user_schema] = make_user_record_batch();
    auto status = add_source(context_, user_table, user_schema);
    if (!status.ok()) {
        fmt::print(stderr, "add source failed {}", status);
        return 1;
    }
    auto [item_batch, item_schema] = make_item_record_batch();
    status = add_source(context_, item_table, item_schema);
    if (!status.ok()) {
        fmt::print(stderr, "add source failed {}", status);
        return 1;
    }

    status = add_join_plan(context_, item_table, user_table, arrow::compute::JoinType::LEFT_OUTER,
                           std::vector({"user_id"s}), std::vector({"user_id"s}));
    if (!status.ok()) {
        fmt::print(stderr, "add join plan failed {}", status);
        return 1;
    }

    status = finish_plan(context_);
    if (!status.ok()) {
        fmt::print(stderr, "finish plan failed {}", status);
        return 1;
    }

    {
        // feed left table, item
        status = feed_input(context_, item_table, item_batch);
        if (!status.ok()) {
            fmt::print(stderr, "feed left failed {}", status);
            return 1;
        }
        // feed right table, user
        status = feed_input(context_, user_table, user_batch);
        if (!status.ok()) {
            fmt::print(stderr, "feed right failed {}", status);
            return 1;
        }
        for (auto node : context_->join_node_->inputs()) {
            context_->join_node_->InputFinished(node, 1);
        }
        // get output record batch
        auto output_result = get_output(context_);
        if (!output_result.ok()) {
            fmt::print(stderr, "get output failed {}", output_result.status());
            return 1;
        }
        auto record_batch = *output_result;
        fmt::print("output batch 1: {}*{}\n", record_batch->num_rows(),
                   record_batch->num_columns());
    }
    {
        // feed left table, item
        status = feed_input(context_, item_table, item_batch);
        if (!status.ok()) {
            fmt::print(stderr, "feed left failed {}", status);
            return 1;
        }
        // feed right table, user
        status = feed_input(context_, user_table, user_batch);
        if (!status.ok()) {
            fmt::print(stderr, "feed right failed {}", status);
            return 1;
        }
        for (auto node : context_->join_node_->inputs()) {
            context_->join_node_->InputFinished(node, 1);
        }
        // get output record batch
        auto output_result = get_output(context_);
        if (!output_result.ok()) {
            fmt::print(stderr, "get output failed {}", output_result.status());
            return 1;
        }
        auto record_batch = *output_result;
        fmt::print("output batch 2: {}*{}\n", record_batch->num_rows(),
                   record_batch->num_columns());
    }

    status = stop(context_);
    if (!status.ok()) {
        fmt::print(stderr, "stop plan failed {}", status);
        return 1;
    }

    return 0;
}
