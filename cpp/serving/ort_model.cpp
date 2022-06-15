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
#include <serving/ort_model.h>
#include <serving/utils.h>
#include <serving/gpu_utils.h>

#include <filesystem>

#include <boost/core/demangle.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace metaspore::serving {

class OrtModelGlobal {
  public:
    OrtModelGlobal() : env_() {}

    Ort::Env env_;
};

static OrtModelGlobal &get_ort_model_global() {
    static OrtModelGlobal global;
    return global;
}

class OrtModelContext {
  public:
    OrtModelContext() : run_options_(), session_options_(), session_(nullptr) {}
    Ort::RunOptions run_options_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string dir_path_;
    std::vector<std::string> input_names_s_;
    std::vector<std::string> output_names_s_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;

    ~OrtModelContext() {
        for (auto p : input_names_) {
            ::free((void *)p);
        }
        for (auto p : output_names_) {
            ::free((void *)p);
        }
    }
};

OrtModel::OrtModel() : context_(std::make_unique<OrtModelContext>()) {}

OrtModel::OrtModel(OrtModel &&) = default;

// To avoid std::unique_ptr requires complete type for OrtModelContext
OrtModel::~OrtModel() = default;

awaitable_status OrtModel::load(std::string dir_path) {
    auto &tp = Threadpools::get_background_threadpool();
    auto r = co_await boost::asio::co_spawn(
        tp,
        [this, &dir_path]() -> awaitable_status {
            auto dir = std::filesystem::path(dir_path);
            if (!std::filesystem::is_directory(dir)) {
                co_return absl::InvalidArgumentError(
                    fmt::format("{} is not a dir for OrtModel to load", dir_path));
            }
            auto file = dir / "model.onnx";
            if (!std::filesystem::is_regular_file(file)) {
                co_return absl::NotFoundError(
                    fmt::format("model.onnx doesn't exist under {}", dir_path));
            }
            if (GpuHelper::is_gpu_available()) {
              spdlog::info("Use cuda:0");
              OrtSessionOptionsAppendExecutionProvider_CUDA(context_->session_options_, 0);
            }
            context_->session_ =
                Ort::Session(get_ort_model_global().env_, file.c_str(), context_->session_options_);
            const size_t input_count = context_->session_.GetInputCount();
            context_->input_names_.reserve(input_count);
            context_->input_names_s_.reserve(input_count);

            for (size_t i = 0UL; i < input_count; ++i) {
                context_->input_names_.push_back(
                    context_->session_.GetInputName(i, context_->allocator_));
                context_->input_names_s_.push_back(context_->input_names_.back());
            }

            const size_t output_count = context_->session_.GetOutputCount();
            context_->output_names_.reserve(output_count);
            context_->output_names_s_.reserve(output_count);
            for (size_t i = 0UL; i < output_count; ++i) {
                context_->output_names_.push_back(
                    context_->session_.GetOutputName(i, context_->allocator_));
                context_->output_names_s_.push_back(context_->output_names_.back());
            }

            context_->dir_path_ = dir_path;
            spdlog::info("OrtModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         dir_path, fmt::join(context_->input_names_s_, ", "),
                         fmt::join(context_->output_names_s_, ", "));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return r;
}

awaitable_result<std::unique_ptr<OrtModelOutput>>
OrtModel::do_predict(std::unique_ptr<OrtModelInput> input) {
    const size_t input_count = context_->session_.GetInputCount();
    std::vector<Ort::Value> inputs;
    inputs.reserve(input_count);
    for (const auto input_name : context_->input_names_) {
        if (auto input_find = input->inputs.find(input_name); input_find != input->inputs.end()) {
            inputs.push_back(std::move(input_find->second.value));
        } else {
            co_return absl::InvalidArgumentError(
                fmt::format("OrtModel cannot find input named {}", input_name));
        }
    }
    const size_t output_count = context_->session_.GetOutputCount();

    auto outs =
        context_->session_.Run(context_->run_options_, &context_->input_names_[0], &inputs[0],
                               input_count, &context_->output_names_[0], output_count);

    auto output = std::make_unique<OrtModelOutput>();
    for (size_t i = 0; i < output_count; ++i) {
        output->outputs.emplace(std::string(context_->output_names_[i]), std::move(outs[i]));
    }
    co_return output;
}

std::string OrtModel::info() const {
    return fmt::format("onnxruntime model loaded from {}/model.onnx", context_->dir_path_);
}

const std::vector<std::string> &OrtModel::input_names() const { return context_->input_names_s_; }

const std::vector<std::string> &OrtModel::output_names() const { return context_->output_names_s_; }

} // namespace metaspore::serving