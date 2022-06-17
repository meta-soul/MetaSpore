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
#include <serving/converters.h>
#include <serving/ort_model.h>
#include <serving/sparse_embedding_bag_model.h>
#include <serving/sparse_lookup_model.h>
#include <serving/threadpool.h>

namespace metaspore::serving {

using namespace std::string_literals;

class SparseEmbeddingBagModelContext {
  public:
    OrtModel ort_model_;
};

SparseEmbeddingBagModel::SparseEmbeddingBagModel() {
    context_ = std::make_unique<SparseEmbeddingBagModelContext>();
}

SparseEmbeddingBagModel::SparseEmbeddingBagModel(SparseEmbeddingBagModel &&) = default;

SparseEmbeddingBagModel::~SparseEmbeddingBagModel() = default;

awaitable_status SparseEmbeddingBagModel::load(std::string dir_path, GrpcClientContextPool &contexts) {
    auto &tp = Threadpools::get_background_threadpool();
    auto r = co_await boost::asio::co_spawn(
        tp,
        [this, &dir_path, &contexts]() -> awaitable_status {
            auto status = co_await context_->ort_model_.load(dir_path, contexts);
            if (!status.ok()) {
                spdlog::error("Cannot load embedding bag ort model from {}", dir_path);
            }
            co_return status;
        },
        boost::asio::use_awaitable);
    co_return r;
}

awaitable_result<std::unique_ptr<OrtModelOutput>>
SparseEmbeddingBagModel::do_predict(std::unique_ptr<SparseLookupModelOutput> input) {
    // create input(indices) ort tensor
    auto ort_indices = Converter::arrow_to_ort_tensor<int64_t>(*input->indices);
    auto ort_weights = Converter::arrow_to_ort_tensor<float>(*input->values);
    auto ort_offsets = Converter::arrow_to_ort_tensor<int64_t>(*input->offsets);
    int64_t bs_shape = 1L;
    auto ort_batch_size = Ort::Value::CreateTensor<int64_t>(
        Ort::AllocatorWithDefaultOptions().GetInfo(), &(input->batch_size), 1UL, &bs_shape, 1UL);
    auto ort_input = std::make_unique<OrtModelInput>();
    ort_input->inputs.emplace("input"s, OrtModelInput::Value{.value = std::move(ort_indices)});
    ort_input->inputs.emplace("weight"s, OrtModelInput::Value{.value = std::move(ort_weights)});
    ort_input->inputs.emplace("offsets"s, OrtModelInput::Value{.value = std::move(ort_offsets)});
    ort_input->inputs.emplace("batch_size"s,
                              OrtModelInput::Value{.value = std::move(ort_batch_size)});
    auto result = co_await context_->ort_model_.do_predict(std::move(ort_input));
    co_return result;
}

std::string SparseEmbeddingBagModel::info() const {
    return "SparseEmbeddingBagModel " + context_->ort_model_.info();
}

const std::vector<std::string> &SparseEmbeddingBagModel::input_names() const {
    return context_->ort_model_.input_names();
}

const std::vector<std::string> &SparseEmbeddingBagModel::output_names() const {
    return context_->ort_model_.output_names();
}

} // namespace metaspore::serving
