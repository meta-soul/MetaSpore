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

#include <common/hashmap/hash_uniquifier.h>
#include <common/logger.h>
#include <serving/inmem_sparse_lookup.h>
#include <serving/sparse_lookup_model.h>
#include <common/threadpool.h>
#include <common/utils.h>

#include <boost/core/demangle.hpp>
#include <fmt/format.h>

#include <filesystem>

namespace metaspore::serving {

using namespace std::string_literals;

class SparseLookupModelGlobal {};

class SparseLookupModelContext {
  public:
    std::unique_ptr<SparseLookupModel::SparseLookupSource> source_;
    std::string dir_path_;
    uint64_t vector_size_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

SparseLookupModel::SparseLookupModel() { context_ = std::make_unique<SparseLookupModelContext>(); }

SparseLookupModel::SparseLookupModel(SparseLookupModel &&) = default;

awaitable_status SparseLookupModel::load(std::string dir_path) {
    auto &tp = Threadpools::get_background_threadpool();
    auto r = co_await boost::asio::co_spawn(
        tp,
        [this, &dir_path]() -> awaitable_status {
            std::filesystem::path p(dir_path);
            if (!std::filesystem::is_directory(p)) {
                co_return absl::NotFoundError(
                    fmt::format("SparseLookupModel cannot find dir {}", dir_path));
            }
            auto embedding_table_dir = p / "embedding_table";
            if (!std::filesystem::is_directory(embedding_table_dir)) {
                co_return absl::NotFoundError(fmt::format(
                    "SparseLookupModel cannot find embedding table dir {}", embedding_table_dir));
            }
            context_->source_ = InMemorySparseLookupSource::make();
            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                context_->source_->load(embedding_table_dir.string()));
            context_->dir_path_ = dir_path;

            CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto size_result,
                                                 context_->source_->get_vector_size());
            if (size_result == 0) {
                co_return absl::InternalError(fmt::format(
                    "SparseLookupModel loaded from {} but got vector size 0", dir_path));
            }

            context_->vector_size_ = size_result;
            context_->inputs_.push_back(fmt::format("{}_fe", p.filename().string()));
            context_->outputs_.push_back(fmt::format("{}_embedding", p.filename().string()));
            spdlog::info("SparseLookupModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         dir_path, fmt::join(context_->inputs_, ", "),
                         fmt::join(context_->outputs_, ", "));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return r;
}

awaitable_result<std::unique_ptr<SparseLookupModelOutput>>
SparseLookupModel::do_predict(std::unique_ptr<SparseLookupModelInput> input) {
    // compute uniquified keys and pull them from lookup source.
    // indices would be modified inplace
    auto keys =
        HashUniquifier::Uniquify((uint64_t *)input->indices->raw_data(), input->indices->size());
    ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
        auto keys_tensor,
        arrow::UInt64Tensor::Make(arrow::Buffer::Wrap(keys), {(int64_t)keys.size()}));
    CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto result, context_->source_->lookup(keys_tensor));
    auto output = std::make_unique<SparseLookupModelOutput>();
    output->values = result;
    output->indices_holder = std::move(input->indices_holder);
    output->offsets_holder = std::move(input->offsets_holder);
    output->indices = input->indices;
    output->offsets = input->offsets;
    output->keys = std::move(keys);
    output->batch_size = input->batch_size;
    co_return output;
}

std::string SparseLookupModel::info() const {
    return fmt::format("SparseLookup model loaded from {}", context_->dir_path_);
}

const std::vector<std::string> &SparseLookupModel::input_names() const { return context_->inputs_; }

const std::vector<std::string> &SparseLookupModel::output_names() const {
    return context_->outputs_;
}

awaitable_result<uint64_t> SparseLookupModel::get_vector_size() {
    co_return context_->vector_size_;
}

SparseLookupModel::~SparseLookupModel() {}

} // namespace metaspore::serving