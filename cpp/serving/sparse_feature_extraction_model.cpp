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

#include <common/logger.h>
#include <common/features/feature_compute_exec.h>
#include <common/features/schema_parser.h>
#include <serving/sparse_feature_extraction_model.h>
#include <common/threadpool.h>
#include <common/utils.h>

#include <boost/core/demangle.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>

namespace metaspore::serving {

class SparseFeatureExtractionModelContext {
  public:
    FeatureComputeExec exec;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

SparseFeatureExtractionModel::SparseFeatureExtractionModel() {
    context_ = std::make_unique<SparseFeatureExtractionModelContext>();
}

SparseFeatureExtractionModel::SparseFeatureExtractionModel(SparseFeatureExtractionModel &&) =
    default;

SparseFeatureExtractionModel::~SparseFeatureExtractionModel() = default;

awaitable_status SparseFeatureExtractionModel::load(std::string dir_path) {
    auto s = co_await boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [this, &dir_path]() -> awaitable_status {
            std::filesystem::path p(dir_path);
            std::filesystem::path schema_file = p / "combine_schema.txt";
            if (!std::filesystem::is_regular_file(schema_file)) {
                co_return absl::NotFoundError(fmt::format(
                    "SparseFeatureExtractionModel cannot find {}", schema_file.string()));
            }
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
                FeatureSchemaParser::parse(schema_file.string(), context_->exec));

            context_->inputs_ = context_->exec.get_input_names();
            context_->outputs_.push_back(
                fmt::format("{}_fe", schema_file.parent_path().filename().string()));
            spdlog::info("SparseFeatureExtractionModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         dir_path, fmt::join(context_->inputs_, ", "),
                         fmt::join(context_->outputs_, ", "));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return s;
}

awaitable_result<std::unique_ptr<SparseFeatureExtractionModelOutput>>
SparseFeatureExtractionModel::do_predict(std::unique_ptr<FeatureExtractionModelInput> input) {
    ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto ctx, context_->exec.start_plan());
    for (const auto &[name, batch] : input->feature_tables) {
        CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
            context_->exec.set_input_schema(ctx, name, batch->schema()));
    }

    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(context_->exec.build_plan(ctx));

    for (const auto &[name, batch] : input->feature_tables) {
        CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(context_->exec.feed_input(ctx, name, batch));
    }

    Defer _([&] { (void)context_->exec.finish_plan(ctx); });

    CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto output_result, context_->exec.execute(ctx));

    auto output = std::make_unique<SparseFeatureExtractionModelOutput>();
    output->values = output_result;
    co_return output;
}

std::string SparseFeatureExtractionModel::info() const { return ""; }

const std::vector<std::string> &SparseFeatureExtractionModel::input_names() const {
    return context_->inputs_;
}

const std::vector<std::string> &SparseFeatureExtractionModel::output_names() const {
    return context_->outputs_;
}

} // namespace metaspore::serving
