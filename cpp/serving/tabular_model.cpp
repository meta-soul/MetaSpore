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
#include <serving/dense_feature_extraction_model.h>
#include <serving/feature_extraction_model_input.h>
#include <serving/ort_model.h>
#include <serving/sparse_embedding_bag_model.h>
#include <serving/sparse_feature_extraction_model.h>
#include <serving/sparse_lookup_model.h>
#include <serving/tabular_model.h>
#include <serving/threadpool.h>

#include <filesystem>

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>

namespace metaspore::serving {

namespace fs = std::filesystem;

struct SparseModelUnit {
    SparseFeatureExtractionModel fe_model;
    SparseLookupModel lookup_model;
    SparseEmbeddingBagModel emb_model;
    std::unique_ptr<Converter> fe_to_lookup_converter;
};

class TabularModelContext {
  public:
    std::vector<SparseModelUnit> sparse_models;
    DenseFeatureExtractionModel dense_model;
    std::unique_ptr<Converter> dense_fe_to_ort_converter;
    OrtModel ort_model;
    // inputs of crt model is unique set of inputs of all Fe models
    std::vector<std::string> inputs_;
};

TabularModel::TabularModel() { context_ = std::make_unique<TabularModelContext>(); }

TabularModel::TabularModel(TabularModel &&) = default;

TabularModel::~TabularModel() = default;

awaitable_status TabularModel::load(std::string dir_path) {
    auto s = co_await boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [&]() -> awaitable_status {
            // load a ctr model
            // 1. find all subdirs prefixed with "sparse_" and load fe/lookup models in them
            fs::path root_dir(dir_path);
            if (!fs::is_directory(root_dir)) {
                co_return absl::NotFoundError(
                    fmt::format("TabularModel cannot find dir {}", dir_path));
            }
            bool dense_loaded = false;

            for (auto const &dir_entry : fs::directory_iterator{root_dir}) {
                if (!dir_entry.is_directory()) {
                    // find if dense schema exist and load it
                    if (dir_entry.path().filename() == "dense_schema.txt") {
                        CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                            context_->dense_model.load(root_dir.string()));
                        context_->dense_fe_to_ort_converter =
                            std::make_unique<DenseFEToOrtConverter>(
                                context_->dense_model.input_names());
                        std::copy(context_->dense_model.input_names().begin(),
                                  context_->dense_model.input_names().end(),
                                  std::back_inserter(context_->inputs_));
                    }
                    continue;
                }

                auto dir_name = dir_entry.path().filename();
                // begin to load components of sparse models
                if (boost::contains(dir_name.string(), "sparse")) {
                    SparseModelUnit unit;
                    int component_loaded = 0;
                    // load sparse model
                    for (auto const &sparse_dir_entry : fs::directory_iterator{dir_entry}) {
                        spdlog::info("find path {} under {}", sparse_dir_entry.path().string(),
                                     dir_entry.path().string());
                        if (!sparse_dir_entry.is_directory() &&
                            sparse_dir_entry.path().filename() == "combine_schema.txt") {
                            spdlog::info("Loading sparse fe model from {}",
                                         dir_entry.path().string());
                            // load sparse fe model
                            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                                unit.fe_model.load(dir_entry.path().string()));
                            component_loaded ^= 0b1;
                            std::copy(unit.fe_model.input_names().begin(),
                                      unit.fe_model.input_names().end(),
                                      std::back_inserter(context_->inputs_));
                        } else if (sparse_dir_entry.path().filename().string() ==
                                   "embedding_table") {
                            spdlog::info("Loading sparse lookup model from {}",
                                         dir_entry.path().string());
                            // load lookup model
                            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                                unit.lookup_model.load(dir_entry.path().string()));
                            unit.fe_to_lookup_converter =
                                std::make_unique<SparseFEToLookupConverter>();
                            component_loaded ^= 0b10;
                        } else if (!sparse_dir_entry.is_directory() &&
                                   sparse_dir_entry.path().filename() == "model.onnx") {
                            spdlog::info("Loading sparse embedding bag model from {}",
                                         dir_entry.path().string());
                            // load sparse emebdding bag model
                            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                                unit.emb_model.load(dir_entry.path().string()));
                            component_loaded ^= 0b100;
                        }
                    }
                    if (component_loaded != 0b111) {
                        co_return absl::NotFoundError(
                            fmt::format("TabularModel with a sparse component under {} requires a "
                                        "combine_schema.txt file, an embedding_table dir and a "
                                        "model.onnx file to initialize, component loaded {:#b}",
                                        dir_name.string(), component_loaded));
                    }
                    context_->sparse_models.emplace_back(std::move(unit));
                } else if (boost::contains(dir_name.string(), "dense")) {
                    // load dense ort model
                    if (!dense_loaded) {
                        CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                            context_->ort_model.load(dir_entry.path()));
                        dense_loaded = true;
                    } else {
                        spdlog::error("TabularModel cannot support more than one dense model");
                        co_return absl::UnimplementedError(
                            "TabularModel cannot support more than one dense model");
                    }
                }
            }

            if (context_->sparse_models.empty() && !context_->dense_fe_to_ort_converter) {
                auto msg = fmt::format(
                    "TabularModel requires at least one fe model while loading from {}", dir_path);
                spdlog::error(msg);
                co_return absl::NotFoundError(msg);
            }

            if (!dense_loaded) {
                auto msg =
                    fmt::format("TabularModel requires an onnx model under {}/dense/", dir_path);
                spdlog::error(msg);
                co_return absl::NotFoundError(msg);
            }

            // get unique input names from all sparse/fe models as the inputs of TabularModel
            std::sort(context_->inputs_.begin(), context_->inputs_.end());
            auto last = std::unique(context_->inputs_.begin(), context_->inputs_.end());
            context_->inputs_.erase(last, context_->inputs_.end());

            spdlog::info("TabularModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         dir_path, fmt::join(context_->inputs_, ", "),
                         fmt::join(this->output_names(), ", "));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return s;
}

std::unique_ptr<FeatureExtractionModelInput> get_input(ModelBase &fe_model,
                                                       const FeatureExtractionModelInput *input) {
    auto out = std::make_unique<FeatureExtractionModelInput>();
    for (const auto &name : fe_model.input_names()) {
        auto find = input->feature_tables.find(name);
        if (find == input->feature_tables.end()) {
            spdlog::error("Fe model required input {} not found");
            return nullptr;
        }
        out->feature_tables[name] = find->second;
    }
    return out;
}

awaitable_result<std::unique_ptr<OrtModelOutput>>
TabularModel::do_predict(std::unique_ptr<FeatureExtractionModelInput> input) {
    // firstly execute sparse fe and lookup
    // set all output to ort input
    auto *fe_input = input.get();
    auto ort_in = std::make_unique<OrtModelInput>();
    for (auto &unit : context_->sparse_models) {
        auto sub_input = get_input(unit.fe_model, fe_input);
        if (!sub_input) {
            co_return absl::NotFoundError("Input not found for sparse fe model");
        }
        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto fe_result,
                                             unit.fe_model.do_predict(std::move(sub_input)));
        auto lookup_in = std::make_unique<SparseLookupModelInput>();
        CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
            unit.fe_to_lookup_converter->convert_input(std::move(fe_result), lookup_in.get()));

        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto lookup_result,
                                             unit.lookup_model.do_predict(std::move(lookup_in)));

        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto emb_ort_result,
                                             unit.emb_model.do_predict(std::move(lookup_result)));

        // merge embedding bag output to ort_in
        for (auto &[name, v] : emb_ort_result->outputs) {
            if (!ort_in->inputs.emplace(name, OrtModelInput::Value{.value = std::move(v)}).second) {
                co_return absl::AlreadyExistsError(
                    fmt::format("Sparse Embedding produced duplicated output {}", name));
            }
        }
    }

    // secondly execute dense fe if it exists
    // and set all output to ort input
    if (context_->dense_fe_to_ort_converter) {
        auto sub_input = get_input(context_->dense_model, fe_input);
        if (!sub_input) {
            co_return absl::NotFoundError("Input not found for dense fe model");
        }
        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
            auto fe_result, context_->dense_model.do_predict(std::move(sub_input)));
        CO_RETURN_IF_STATUS_NOT_OK(
            context_->dense_fe_to_ort_converter->convert_input(std::move(fe_result), ort_in.get()));
    }

    // finally execute ort model prediction
    CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto final_result,
                                         context_->ort_model.do_predict(std::move(ort_in)));
    co_return final_result;
}

std::string TabularModel::info() const { return ""; }

const std::vector<std::string> &TabularModel::input_names() const { return context_->inputs_; }

const std::vector<std::string> &TabularModel::output_names() const {
    return context_->ort_model.output_names();
}

} // namespace metaspore::serving