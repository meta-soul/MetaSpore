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
#include <serving/py_preprocessing_model.h>
#include <serving/py_preprocessing_ort_model.h>
#include <common/threadpool.h>

#include <filesystem>

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>

namespace metaspore::serving {

namespace fs = std::filesystem;

class PyPreprocessingOrtModelContext {
  public:
    PyPreprocessingModel preprocessing_model;
    OrtModel ort_model;
    std::unique_ptr<Converter> preprocessing_to_ort_converter;
};

PyPreprocessingOrtModel::PyPreprocessingOrtModel() {
    context_ = std::make_unique<PyPreprocessingOrtModelContext>();
}

PyPreprocessingOrtModel::PyPreprocessingOrtModel(PyPreprocessingOrtModel &&) = default;

PyPreprocessingOrtModel::~PyPreprocessingOrtModel() = default;

awaitable_status PyPreprocessingOrtModel::load(std::string dir_path) {
    auto &tp = Threadpools::get_background_threadpool();
    auto s = co_await boost::asio::co_spawn(
        tp,
        [this, &dir_path]() -> awaitable_status {
            auto dir = std::filesystem::path(dir_path);
            if (!std::filesystem::is_directory(dir)) {
                co_return absl::InvalidArgumentError(
                    fmt::format("{} is not a dir for PyPreprocessingOrtModel to load", dir_path));
            }
            auto preprocess_dir = dir / "preprocess";
            if (!std::filesystem::is_directory(preprocess_dir)) {
                co_return absl::InvalidArgumentError(
                    fmt::format("{} is not a dir for PyPreprocessingModel to load", preprocess_dir.string()));
            }
            auto preprocessor_py = preprocess_dir / "preprocessor.py";
            if (!std::filesystem::is_regular_file(preprocessor_py)) {
                co_return absl::NotFoundError(
                    fmt::format("preprocessor.py doesn't exist under {}", preprocess_dir.string()));
            }
            auto main_dir = dir / "main";
            if (!std::filesystem::is_directory(main_dir)) {
                co_return absl::InvalidArgumentError(
                    fmt::format("{} is not a dir for OrtModel to load", main_dir.string()));
            }
            auto model_onnx = main_dir / "model.onnx";
            if (!std::filesystem::is_regular_file(model_onnx)) {
                co_return absl::NotFoundError(
                    fmt::format("model.onnx doesn't exist under {}", main_dir.string()));
            }
            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                context_->preprocessing_model.load(preprocess_dir.string()));
            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                context_->ort_model.load(main_dir.string()));
            auto preprocess_out = context_->preprocessing_model.output_names();
            auto main_in = context_->ort_model.input_names();
            std::unordered_set<std::string> out_set(preprocess_out.begin(), preprocess_out.end());
            std::unordered_set<std::string> in_set(main_in.begin(), main_in.end());
            if (out_set != in_set) {
                co_return absl::InvalidArgumentError(
                    fmt::format("fail to load PyPreprocessingModel from {}; "
                                "preprocess_out [{}] and main_in [{}] mismatch",
                                dir_path,
                                fmt::join(preprocess_out, ", "),
                                fmt::join(main_in, ", ")));
            }
            context_->preprocessing_to_ort_converter =
                std::make_unique<PyPreprocessingToOrtConverter>(main_in);
            spdlog::info("PyPreprocessingOrtModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         dir_path,
                         fmt::join(input_names(), ", "),
                         fmt::join(output_names(), ", "));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return s;
}

awaitable_result<std::unique_ptr<OrtModelOutput>>
PyPreprocessingOrtModel::do_predict(std::unique_ptr<PyPreprocessingModelInput> input) {
    CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
        auto preprocess_output, context_->preprocessing_model.do_predict(std::move(input)));
    auto main_input = std::make_unique<OrtModelInput>();
    CO_RETURN_IF_STATUS_NOT_OK(
        context_->preprocessing_to_ort_converter->convert_input(
            std::move(preprocess_output), main_input.get()));
    CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
        auto final_result, context_->ort_model.do_predict(std::move(main_input)));
    co_return final_result;
}

std::string PyPreprocessingOrtModel::info() const { return ""; }

const std::vector<std::string> &PyPreprocessingOrtModel::input_names() const {
    return context_->preprocessing_model.input_names();
}

const std::vector<std::string> &PyPreprocessingOrtModel::output_names() const {
    return context_->ort_model.output_names();
}

} // namespace metaspore::serving
