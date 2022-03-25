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

#include <unordered_map>

#include <common/logger.h>
#include <serving/feature_compute_exec.h>
#include <serving/threadpool.h>
#include <serving/utils.h>

#include <boost/asio/experimental/concurrent_channel.hpp>
#include <fmt/format.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/exec/exec_plan.h>
#include <arrow/util/async_generator.h>

namespace cp = arrow::compute;

namespace metaspore::serving {

using namespace std::string_literals;

using channel_type =
    boost::asio::experimental::concurrent_channel<void(boost::system::error_code, cp::ExecBatch)>;

class FeatureComputeContext {
  public:
    struct InputSource {
        arrow::PushGenerator<arrow::util::optional<cp::ExecBatch>> input_queue;
        std::shared_ptr<cp::Declaration> decl;

        InputSource(std::shared_ptr<arrow::Schema> schema)
            : input_queue(), decl(std::make_shared<cp::Declaration>(
                                 "source", cp::SourceNodeOptions{schema, input_queue})) {}
    };

    std::unordered_map<std::string, InputSource> name_source_map_;
    cp::ExecNode *root_node_{nullptr};
    cp::ExecNode *root_before_sink_node_{nullptr};
    std::shared_ptr<cp::Declaration> root_decl_;
    channel_type channel_{Threadpools::get_compute_threadpool(), 10};
    arrow::Future<> sink_future_{arrow::Future<>::Make()};
    std::shared_ptr<cp::ExecPlan> plan_;
};

FeatureComputeExec::FeatureComputeExec() { context_ = std::make_unique<FeatureComputeContext>(); }

FeatureComputeExec::~FeatureComputeExec() {}

FeatureComputeExec::FeatureComputeExec(FeatureComputeExec &&) = default;

status FeatureComputeExec::add_source(const std::string &name) {
    if (name.empty()) {
        return absl::InvalidArgumentError("FeatureComputeExec input name cannot be empty");
    }
    auto pair =
        context_->name_source_map_.emplace(name, FeatureComputeContext::InputSource(nullptr));
    if (!pair.second) {
        return absl::AlreadyExistsError(fmt::format("Input source {} already exists", name));
    }
    context_->root_decl_ = pair.first->second.decl;
    return absl::OkStatus();
}

status FeatureComputeExec::feed_input(const std::string &source_name,
                                      std::shared_ptr<arrow::RecordBatch> batch) {
    auto source = context_->name_source_map_.find(source_name);
    if (source == context_->name_source_map_.end()) {
        return absl::NotFoundError(
            fmt::format("FeatureComputeExec feed_input with non-exist name {}", source_name));
    }
    cp::ExecBatch exec_batch(*batch);
    source->second.input_queue.producer().Push(
        arrow::util::make_optional<cp::ExecBatch>(std::move(exec_batch)));
    return absl::OkStatus();
}

status FeatureComputeExec::set_input_schema(const std::string &source_name,
                                            std::shared_ptr<arrow::Schema> schema) {
    auto find = context_->name_source_map_.find(source_name);
    if (find == context_->name_source_map_.end()) {
        return absl::NotFoundError(
            fmt::format("Cannot find {} when set_input_schema", source_name));
    }
    auto source_option =
        std::static_pointer_cast<cp::SourceNodeOptions>(find->second.decl->options);
    source_option->output_schema = schema;
    return absl::OkStatus();
}

awaitable_result<std::shared_ptr<arrow::RecordBatch>> FeatureComputeExec::execute() {
    finish_join(context_->root_node_);
    cp::ExecBatch exec_batch =
        co_await context_->channel_.async_receive(boost::asio::use_awaitable);
    ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
        auto rb_result,
        exec_batch.ToRecordBatch(context_->root_before_sink_node_->output_schema()));
    co_return rb_result;
}

status FeatureComputeExec::add_join_plan(const std::string &left_source_name,
                                         const std::string &right_source_name,
                                         cp::JoinType join_type,
                                         const std::vector<std::string> &left_key_names,
                                         const std::vector<std::string> &right_key_names) {
    auto left_source = context_->name_source_map_.find(left_source_name);
    if (left_source == context_->name_source_map_.end()) {
        return absl::NotFoundError(
            fmt::format("FeatureComputeExec add join with non-exist name {}", left_source_name));
    }
    auto right_source = context_->name_source_map_.find(right_source_name);
    if (right_source == context_->name_source_map_.end()) {
        return absl::NotFoundError(
            fmt::format("FeatureComputeExec add join with non-exist name {}", right_source_name));
    }

    cp::HashJoinNodeOptions join_opts{
        join_type, left_key_names | ranges::views::transform([](const std::string &name) {
                       return arrow::FieldRef(name);
                   }) | ranges::to<std::vector>(),
        right_key_names | ranges::views::transform([](const std::string &name) {
            return arrow::FieldRef(name);
        }) | ranges::to<std::vector>()};
    cp::Declaration join("hashjoin",
                         {cp::Declaration::Input(*left_source->second.decl),
                          cp::Declaration::Input(*right_source->second.decl)},
                         join_opts, "join_node");
    context_->root_decl_ = std::make_shared<cp::Declaration>(std::move(join));
    return absl::OkStatus();
}

status FeatureComputeExec::add_projection(std::vector<arrow::compute::Expression> expressions) {
    cp::ProjectNodeOptions options(std::move(expressions));
    cp::Declaration project("project", {cp::Declaration::Input(*context_->root_decl_)}, options,
                            "project_node");
    context_->root_decl_ = std::make_shared<cp::Declaration>(std::move(project));
    return absl::OkStatus();
}

struct SinkConsumer : public cp::SinkNodeConsumer {

    SinkConsumer(std::shared_ptr<arrow::Schema> schema, channel_type &ch, arrow::Future<> fut)
        : output_schema_(schema), channel_(ch), fut_(std::move(fut)) {}

    arrow::Status Consume(cp::ExecBatch batch) override {
        bool s = channel_.try_send(boost::system::error_code(), std::move(batch));
        if (!s) {
            return arrow::Status::CapacityError("cannot consume batch and send to channel");
        } else {
            return arrow::Status::OK();
        }
    }

    arrow::Future<> Finish() override { return fut_; }

    std::shared_ptr<arrow::Schema> output_schema_;
    channel_type &channel_;
    arrow::Future<> fut_;
};

status FeatureComputeExec::build_plan() {
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto plan, cp::ExecPlan::Make());
    // create sink reader
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto root_result, context_->root_decl_->AddToPlan(plan.get()));
    context_->root_before_sink_node_ = root_result;
    ASSIGN_RESULT_OR_RETURN_NOT_OK(
        auto sink_result,
        cp::MakeExecNode(
            "consuming_sink", plan.get(), {root_result},
            cp::ConsumingSinkNodeOptions{std::make_shared<SinkConsumer>(
                root_result->output_schema(), context_->channel_, context_->sink_future_)}));
    context_->root_node_ = sink_result;

    // validate the ExecPlan
    CALL_AND_RETURN_IF_STATUS_NOT_OK(plan->Validate());
    SPDLOG_DEBUG("FeatureComputeExec created plan {}", plan->ToString());
    // start the ExecPlan
    CALL_AND_RETURN_IF_STATUS_NOT_OK(plan->StartProducing());
    context_->plan_ = plan;
    return absl::OkStatus();
}

status FeatureComputeExec::finish_plan() {
    for (auto &[name, queue] : context_->name_source_map_) {
        // to trigger arrow source node finish its async future loop
        queue.input_queue.producer().Push(
            arrow::IterationTraits<arrow::util::optional<cp::ExecBatch>>::End());
    }
    // context_->plan_->StopProducing();
    context_->sink_future_.MarkFinished();
    context_->sink_future_ = arrow::Future<>::Make();
    context_->root_before_sink_node_ = nullptr;
    context_->root_node_ = nullptr;
    auto plan = context_->plan_;
    context_->plan_.reset();
    return ArrowStatusToAbsl::arrow_status_to_absl(plan->finished().status());
}

void FeatureComputeExec::finish_join(arrow::compute::ExecNode *node) {
    for (auto input : node->inputs()) {
        if (input->label() == "join_node"s) {
            for (auto recurse_input : input->inputs()) {
                input->InputFinished(recurse_input, 1);
            }
        }
        finish_join(input);
    }
}

std::vector<std::string> FeatureComputeExec::get_input_names() const {
    std::vector<std::string> v;
    v.reserve(context_->name_source_map_.size());
    for (const auto &[name, s] : context_->name_source_map_) {
        v.push_back(name);
    }
    return v;
}

} // namespace metaspore::serving