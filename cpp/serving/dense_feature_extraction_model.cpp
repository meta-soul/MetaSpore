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
#include <serving/dense_feature_extraction_model.h>
#include <serving/schema_parse.h>
#include <serving/threadpool.h>

#include <arrow/api.h>
#include <boost/core/demangle.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <fstream>

namespace metaspore::serving {

class DenseFeatureExtractionModelContext {
  public:
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

DenseFeatureExtractionModel::DenseFeatureExtractionModel() {
    context_ = std::make_unique<DenseFeatureExtractionModelContext>();
}

DenseFeatureExtractionModel::DenseFeatureExtractionModel(DenseFeatureExtractionModel &&) = default;

DenseFeatureExtractionModel::~DenseFeatureExtractionModel() = default;

awaitable_status DenseFeatureExtractionModel::load(std::string dir_path) {
    auto s = co_await boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [=]() -> awaitable_status {
            namespace fs = std::filesystem;
            for (const auto &entry : fs::directory_iterator(dir_path)) {
                if (entry.is_regular_file() && entry.path().filename() == "dense_schema.txt") {
                    // parse dense schema with table infos
                    std::ifstream ifs(entry.path().c_str());
                    std::string line;
                    while (std::getline(ifs, line)) {
                        auto table_name = FeatureSchemaParser::parse_table_name_from_config(line);
                        if (table_name.empty()) {
                            continue;
                        }
                        context_->inputs_.push_back(table_name);
                        context_->outputs_.push_back(table_name);
                    }
                    if (context_->inputs_.empty()) {
                        co_return absl::InvalidArgumentError(
                            fmt::format("Cannot find table config from {}", entry.path().string()));
                    }
                    break;
                }
            }
            if (context_->inputs_.empty()) {
                co_return absl::NotFoundError(
                    fmt::format("Cannot find any input table from {}/dense_schema.txt", dir_path));
            }
            spdlog::info("DenseFeatureExtractionModel loaded from {} with inputs [{}] and "
                         "producing outputs [{}]",
                         dir_path, fmt::join(context_->inputs_, ","),
                         fmt::join(context_->outputs_, ","));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return s;
}

awaitable_result<std::unique_ptr<DenseFeatureExtractionModelOutput>>
DenseFeatureExtractionModel::do_predict(std::unique_ptr<FeatureExtractionModelInput> input) {
    // TODO: support expression projection for dense features
    // For now we just convert all columns of a batch to FloatTensor
    auto output = std::make_unique<DenseFeatureExtractionModelOutput>();
    for (const auto &[name, batch] : input->feature_tables) {
        // convert a batch of floats to float tensor
        int64_t rows = batch->num_rows();
        int64_t cols = batch->num_columns();
        std::vector<const float *> columns;
        columns.reserve(cols);
        for (int j = 0; j < cols; ++j) {
            auto col = batch->column(j);
            auto float_array = std::dynamic_pointer_cast<arrow::FloatArray>(col);
            if (!float_array) {
                co_return absl::InvalidArgumentError(fmt::format(
                    "The {}th array in feature table {} should be float array, but got {}", j, name,
                    col->type()->ToString()));
            }
            columns.push_back(float_array->raw_values());
        }

        ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto buffer,
                                          arrow::AllocateBuffer(rows * cols * sizeof(float)));
        ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
            auto tensor, arrow::FloatTensor::Make(std::shared_ptr<arrow::Buffer>(buffer.release()),
                                                  {rows, cols}));
        float *f = (float *)tensor->raw_mutable_data();

        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                f[i * cols + j] = columns[j][i];
            }
        }

        output->feature_tensors[name] = tensor;
    }
    co_return output;
}

std::string DenseFeatureExtractionModel::info() const { return ""; }

const std::vector<std::string> &DenseFeatureExtractionModel::input_names() const {
    return context_->inputs_;
}

const std::vector<std::string> &DenseFeatureExtractionModel::output_names() const {
    return context_->outputs_;
}

} // namespace metaspore::serving