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
#include <serving/model_manager.h>
#include <serving/ort_model.h>
#include <serving/sparse_feature_extraction_model.h>
#include <serving/tabular_model.h>
#include <serving/threadpool.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <filesystem>

namespace metaspore::serving {

void ModelManager::init(const std::string &dir_path) {
    // scan all subdirs and try to load them
    namespace fs = std::filesystem;
    auto d = fs::path(dir_path);
    spdlog::info("Scan {} and load model", fs::absolute(d));
    for (const auto &dir_entry : fs::directory_iterator(d)) {
        if (dir_entry.is_directory()) {
            boost::asio::co_spawn(
                Threadpools::get_background_threadpool(),
                [=]() -> awaitable<void> {
                    auto sub_dir = dir_entry.path();
                    auto name = dir_entry.path().filename();
                    spdlog::info("Try to load model from {} with name {} during init", sub_dir,
                                 name);
                    auto s = co_await load(dir_entry.path(), dir_entry.path().filename());
                    if (!s.ok()) {
                        spdlog::info("Load model from {} during init failed {}, ignored", sub_dir,
                                     s);
                    }
                },
                boost::asio::detached);
        }
    }
    // wait until all loading tasks finished
    Threadpools::get_background_threadpool().join();
}

awaitable_status ModelManager::load(const std::string &dir_path, const std::string &name) {
    auto s = co_await boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [=]() -> awaitable_status {
            TabularModel model;
            auto status = co_await model.load(dir_path);
            if (!status.ok()) {
                spdlog::error("Cannot load TabularModel {} from {}: {}", name, dir_path, status);
                co_return status;
            }
            auto runner = std::make_shared<GrpcTabularModelRunner>();
            runner->model = std::make_unique<TabularModel>(std::move(model));
            runner->input_conveter =
                std::make_unique<GrpcRequestToFEConverter>(runner->model->input_names());
            runner->output_conveter =
                std::make_unique<OrtToGrpcReplyConverter>(runner->model->output_names());
            spdlog::info("Loaded model {} from {}", name, dir_path);
            std::unique_lock wl(mu_);
            models_[name] = runner;
            co_return absl::OkStatus();
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
