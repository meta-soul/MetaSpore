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
#include <serving/feature_extraction_model_input.h>
#include <serving/model_manager.h>
#include <serving/tabular_model.h>
#include <serving/py_preprocessing_model.h>
#include <serving/py_preprocessing_ort_model.h>
#include <serving/ort_model.h>
#include <common/threadpool.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <future>
#include <vector>
#include <filesystem>
#include <boost/asio/use_future.hpp>

namespace metaspore::serving {

void ModelManager::init(const std::string &dir_path) {
    // scan all subdirs and try to load them
    namespace fs = std::filesystem;
    auto d = fs::path(dir_path);
    spdlog::info("Scan {} and load model", fs::absolute(d));
    std::vector<std::future<void>> futures;
    for (const auto &dir_entry : fs::directory_iterator(d)) {
        if (dir_entry.is_directory()) {
            std::future<void> future = boost::asio::co_spawn(
                                           Threadpools::get_background_threadpool(),
                                           [this, dir_entry]() -> awaitable<void> {
                                               auto sub_dir = dir_entry.path();
                                               auto name = dir_entry.path().filename();
                                               spdlog::info("ModelManager: Try to load model from {} with name {} during init",
                                                            sub_dir, name);
                                               auto s = co_await load(dir_entry.path(), dir_entry.path().filename());
                                               if (!s.ok()) {
                                                   spdlog::info(
                                                       "ModelManager: Load model from {} during init failed {}, ignored",
                                                       sub_dir, s);
                                               }
                                           },
                                           boost::asio::use_future);
            futures.push_back(std::move(future));
        }
    }
    // wait until all loading tasks finished
    for (size_t i = 0; i < futures.size(); i++)
        futures.at(i).get();
}

awaitable_status ModelManager::load(const std::string &dir_path, const std::string &name) {
    auto s = co_await boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [this, &dir_path, &name]() -> awaitable_status {
            // load order: tabular, preproc_ort, ort
            auto load_tabular_fn = [this, &dir_path, &name]() -> awaitable_status {
                TabularModel model;
                auto status = co_await model.load(dir_path);
                if (!status.ok()) {
                    spdlog::error("ModelManager: Cannot load TabularModel {} from {}: {}", name,
                                  dir_path, status);
                    co_return status;
                }
                auto runner = std::make_shared<GrpcTabularModelRunner>();
                runner->model = std::make_unique<TabularModel>(std::move(model));
                runner->input_conveter =
                    std::make_unique<GrpcRequestToFEConverter>(runner->model->input_names());
                runner->output_conveter =
                    std::make_unique<OrtToGrpcReplyConverter>(runner->model->output_names());
                spdlog::info("ModelManager: Loaded TabularModel {} from {}", name, dir_path);
                std::unique_lock wl(mu_);
                models_[name] = runner;
                co_return absl::OkStatus();
            };
            auto load_preproc_ort_fn = [this, &dir_path, &name]() -> awaitable_status {
                PyPreprocessingOrtModel model;
                auto status = co_await model.load(dir_path);
                if (!status.ok()) {
                    spdlog::error("ModelManager: Cannot load PyPreprocessingOrtModel {} from {}: {}", name,
                                  dir_path, status);
                    co_return status;
                }
                auto runner = std::make_shared<GrpcPreprocessingOrtModelRunner>();
                runner->model = std::make_unique<PyPreprocessingOrtModel>(std::move(model));
                runner->input_conveter =
                    std::make_unique<GrpcRequestToPyPreprocessingConverter>(runner->model->input_names());
                runner->output_conveter =
                    std::make_unique<OrtToGrpcReplyConverter>(runner->model->output_names());
                spdlog::info("ModelManager: Loaded PyPreprocessingOrtModel {} from {}", name, dir_path);
                std::unique_lock wl(mu_);
                models_[name] = runner;
                co_return absl::OkStatus();
            };
            auto load_ort_fn = [this, &dir_path, &name]() -> awaitable_status {
                OrtModel model;
                auto status = co_await model.load(dir_path);
                if (!status.ok()) {
                    spdlog::error("ModelManager: Cannot load OrtModel {} from {}: {}", name,
                                  dir_path, status);
                    co_return status;
                }
                auto runner = std::make_shared<GrpcOrtModelRunner>();
                runner->model = std::make_unique<OrtModel>(std::move(model));
                runner->input_conveter =
                    std::make_unique<GrpcRequestToOrtConverter>(runner->model->input_names());
                runner->output_conveter =
                    std::make_unique<OrtToGrpcReplyConverter>(runner->model->output_names());
                spdlog::info("ModelManager: Loaded OrtModel {} from {}", name, dir_path);
                std::unique_lock wl(mu_);
                models_[name] = runner;
                co_return absl::OkStatus();
            };

            auto status = co_await load_tabular_fn();
            if (status.ok())
                co_return status;
            status = co_await load_preproc_ort_fn();
            if (status.ok())
                co_return status;
            status = co_await load_ort_fn();
            if (status.ok())
                co_return status;
            spdlog::error("ModelManager: Cannot load model {} from {}: {}", name,
                          dir_path, status);
            co_return status;
        },
        boost::asio::use_awaitable);
    co_return s;
}

result<std::shared_ptr<GrpcModelRunner>> ModelManager::get_model(const std::string &name) {
    std::shared_lock rl(mu_);
    if (auto find = models_.find(name); find != models_.end()) {
        return find->second;
    } else {
        return absl::NotFoundError(fmt::format("Cannot find model {}", name));
    }
}

} // namespace metaspore::serving
