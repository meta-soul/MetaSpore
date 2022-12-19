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
#include <common/features/feature_compute_exec.h>
#include <common/threadpool.h>
#include <common/utils.h>

#include <boost/asio/experimental/concurrent_channel.hpp>
#include <fmt/format.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/exec/exec_plan.h>
#include <arrow/util/async_generator.h>

namespace cp = arrow::compute;

namespace metaspore {

using namespace std::string_literals;

using channel_type =
    boost::asio::experimental::concurrent_channel<void(boost::system::error_code, cp::ExecBatch)>;

class MSSourceNodeOptions : public cp::SourceNodeOptions {
  public:
    MSSourceNodeOptions(
        const std::string &name_, std::shared_ptr<arrow::Schema> output_schema,
        std::function<arrow::Future<arrow::util::optional<cp::ExecBatch>>()> generator)
        : cp::SourceNodeOptions(output_schema, generator), name(name_) {}
    std::string name;
};

class FeatureComputeExecContext {
  public:
    std::unordered_map<std::string, std::shared_ptr<MSSourceNodeOptions>> name_source_map_;
    std::unordered_map<std::string, arrow::PushGenerator<arrow::util::optional<cp::ExecBatch>>>
        name_gen_map_;
    std::shared_ptr<cp::Declaration> root_decl_;
    cp::ExecNode *root_node_{nullptr};
    cp::ExecNode *root_before_sink_node_{nullptr};
    channel_type channel_{Threadpools::get_compute_threadpool(), 10};
    arrow::Future<> sink_future_{arrow::Future<>::Make()};
    std::shared_ptr<cp::ExecPlan> plan_;
};

class FeatureComputeContext {
  public:
    std::unordered_map<std::string, cp::Declaration> name_source_map_;
    std::shared_ptr<cp::Declaration> root_decl_;
};

FeatureComputeExec::FeatureComputeExec() { context_ = std::make_unique<FeatureComputeContext>(); }

FeatureComputeExec::~FeatureComputeExec() {}

FeatureComputeExec::FeatureComputeExec(FeatureComputeExec &&) = default;

status FeatureComputeExec::add_source(const std::string &name) {
    if (name.empty()) {
        return absl::InvalidArgumentError("FeatureComputeExec input name cannot be empty");
    }
    if (context_->name_source_map_.find(name) != context_->name_source_map_.end()) {
        return absl::AlreadyExistsError(fmt::format("Input source {} already exists", name));
    }
    auto pair = context_->name_source_map_.emplace(
        name, cp::Declaration{"source", MSSourceNodeOptions{name, nullptr, nullptr}});
    context_->root_decl_ = std::make_shared<cp::Declaration>(pair.first->second);
    return absl::OkStatus();
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
    cp::HashJoinNodeOptions join_opts{
        join_type, left_key_names | ranges::views::transform([](const std::string &name) {
                       return arrow::FieldRef(name);
                   }) | ranges::to<std::vector>(),
        right_key_names | ranges::views::transform([](const std::string &name) {
            return arrow::FieldRef(name);
        }) | ranges::to<std::vector>()};
#pragma GCC diagnostic pop

    cp::Declaration join(
        "hashjoin",
        {cp::Declaration::Input(left_source->second), cp::Declaration::Input(right_source->second)},
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

static void find_source_node(std::shared_ptr<FeatureComputeExecContext> &ctx,
                             cp::Declaration &decl) {
    if (decl.factory_name == "source") {
        auto source_option = std::static_pointer_cast<MSSourceNodeOptions>(decl.options);
        // decl.options is a shared pointer, we need to make a deep copy
        auto source_option_copied = std::make_shared<MSSourceNodeOptions>(*source_option);
        decl.options = source_option_copied;
        ctx->name_source_map_.emplace(source_option->name, source_option_copied);
    }
    for (auto &input : decl.inputs) {
        find_source_node(ctx, arrow::util::get<cp::Declaration>(input));
    }
}

result<std::shared_ptr<FeatureComputeExecContext>> FeatureComputeExec::start_plan() const {
    auto ctx = std::make_shared<FeatureComputeExecContext>();
    ctx->root_decl_ = std::make_shared<cp::Declaration>(*context_->root_decl_);
    find_source_node(ctx, *ctx->root_decl_);
    return ctx;
}

status FeatureComputeExec::set_input_schema(std::shared_ptr<FeatureComputeExecContext> &ctx,
                                            const std::string &source_name,
                                            std::shared_ptr<arrow::Schema> schema) const {
    auto find = ctx->name_source_map_.find(source_name);
    if (find == ctx->name_source_map_.end()) {
        return absl::NotFoundError(
            fmt::format("FeatureComputeExec set_input_schema with non-exist name {}", source_name));
    }

    // set schema when the first input arrived
    auto source_option = find->second;
    source_option->output_schema = schema;
    auto pair = ctx->name_gen_map_.emplace(
        source_name, arrow::PushGenerator<arrow::util::optional<cp::ExecBatch>>{});
    source_option->generator = pair.first->second;
    return absl::OkStatus();
}

status FeatureComputeExec::feed_input(std::shared_ptr<FeatureComputeExecContext> &ctx,
                                      const std::string &source_name,
                                      std::shared_ptr<arrow::RecordBatch> batch) const {
    auto source = ctx->name_gen_map_.find(source_name);
    if (source == ctx->name_gen_map_.end()) {
        return absl::NotFoundError(
            fmt::format("FeatureComputeExec feed_input with non-exist name {}", source_name));
    }
    cp::ExecBatch exec_batch(*batch);
    source->second.producer().Push(
        arrow::util::make_optional<cp::ExecBatch>(std::move(exec_batch)));
    return absl::OkStatus();
}

awaitable_result<std::shared_ptr<arrow::RecordBatch>>
FeatureComputeExec::execute(std::shared_ptr<FeatureComputeExecContext> &ctx) const {
    finish_join(ctx->root_node_);
    cp::ExecBatch exec_batch = co_await ctx->channel_.async_receive(boost::asio::use_awaitable);
    ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
        auto rb_result, exec_batch.ToRecordBatch(ctx->root_before_sink_node_->output_schema()));
    co_return rb_result;
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

status FeatureComputeExec::build_plan(std::shared_ptr<FeatureComputeExecContext> &ctx) const {
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto plan, cp::ExecPlan::Make());
    // create sink reader
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto root_result, ctx->root_decl_->AddToPlan(plan.get()));
    ctx->root_before_sink_node_ = root_result;
    ASSIGN_RESULT_OR_RETURN_NOT_OK(
        auto sink_result,
        cp::MakeExecNode("consuming_sink", plan.get(), {root_result},
                         cp::ConsumingSinkNodeOptions{std::make_shared<SinkConsumer>(
                             root_result->output_schema(), ctx->channel_, ctx->sink_future_)}));
    ctx->root_node_ = sink_result;

    // validate the ExecPlan
    CALL_AND_RETURN_IF_STATUS_NOT_OK(plan->Validate());
    SPDLOG_DEBUG("FeatureComputeExec created plan {}", plan->ToString());
    // start the ExecPlan
    CALL_AND_RETURN_IF_STATUS_NOT_OK(plan->StartProducing());
    ctx->plan_ = plan;
    return absl::OkStatus();
}

status FeatureComputeExec::finish_plan(std::shared_ptr<FeatureComputeExecContext> &ctx) const {
    for (auto &[name, queue] : ctx->name_gen_map_) {
        // to trigger arrow source node finish its async future loop
        queue.producer().Push(arrow::IterationTraits<arrow::util::optional<cp::ExecBatch>>::End());
    }
    // context_->plan_->StopProducing();
    ctx->sink_future_.MarkFinished();
    ctx->sink_future_ = arrow::Future<>::Make();
    ctx->root_before_sink_node_ = nullptr;
    ctx->root_node_ = nullptr;
    auto plan = ctx->plan_;
    ctx->plan_.reset();
    return ArrowStatusToAbsl::arrow_status_to_absl(plan->finished().status());
}

void FeatureComputeExec::finish_join(arrow::compute::ExecNode *node) const {
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

} // namespace metaspore
