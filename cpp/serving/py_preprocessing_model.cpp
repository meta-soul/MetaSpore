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

#include <filesystem>
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/process/search_path.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <agrpc/asioGrpc.hpp>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <fmt/format.h>
#include <common/logger.h>
#include <serving/threadpool.h>
#include <serving/utils.h>
#include <serving/metaspore.pb.h>
#include <serving/metaspore.grpc.pb.h>
#include <serving/grpc_client_context_pool.h>
#include <serving/py_preprocessing_process.h>
#include <serving/py_preprocessing_model.h>

namespace metaspore::serving {

class PyPreprocessingModelContext {
  public:
    PyPreprocessingProcess process_;
    std::unique_ptr<Predict::Stub> stub_;
    std::filesystem::path temp_dir_;

    ~PyPreprocessingModelContext() {
        if (!temp_dir_.empty())
            std::filesystem::remove_all(temp_dir_);
    }
};

PyPreprocessingModel::PyPreprocessingModel() { context_ = std::make_unique<PyPreprocessingModelContext>(); }

PyPreprocessingModel::PyPreprocessingModel(PyPreprocessingModel &&) = default;

awaitable_status PyPreprocessingModel::load(std::string dir_path) {
    auto &tp = Threadpools::get_background_threadpool();
    auto r = co_await boost::asio::co_spawn(
        tp,
        [this, &dir_path]() -> awaitable_status {
            std::filesystem::path p(dir_path);
            if (!std::filesystem::is_directory(p)) {
                co_return absl::NotFoundError(
                    fmt::format("PyPreprocessingModel cannot find dir {}", dir_path));
            }

            // set Python executable
            auto py_path = boost::process::search_path("python");
            if (py_path.empty()) {
                co_return absl::NotFoundError("PyPreprocessingModel cannot find the Python interpreter");
            }
            context_->process_.set_python_executable(py_path.string());

            // set venv dir
            auto uuid = boost::uuids::to_string(boost::uuids::random_generator()());
            auto temp_dir = std::filesystem::temp_directory_path() / uuid;
            if (!std::filesystem::create_directory(temp_dir)) {
                co_return absl::FailedPreconditionError(
                    fmt::format("PyPreprocessingModel cannot create temp dir {}", temp_dir));
            }
            context_->temp_dir_ = temp_dir;
            auto venv_dir = temp_dir / "venv";
            context_->process_.set_virtual_env_dir(venv_dir.string());

            // set requirement file
            auto req_file = p / "requirements.txt";
            if (std::filesystem::exists(req_file)) {
                context_->process_.set_requirement_file(req_file.string());
            }

            // set service script
            auto prog_dir = boost::dll::program_location().parent_path();
            auto service_script = (prog_dir / "preprocessor_service.py").string();
            if (!std::filesystem::exists(service_script)) {
                co_return absl::NotFoundError(fmt::format(
                    "PyPreprocessingModel cannot find preprocessor service script {}", service_script));
            }
            context_->process_.set_service_script_file(service_script);

            // set preprocessor config dir
            auto preprocessor_script = p / "preprocessor.py";
            if (!std::filesystem::exists(preprocessor_script)) {
                co_return absl::NotFoundError(fmt::format(
                    "PyPreprocessingModel cannot find preprocessor script {}", preprocessor_script));
            }
            context_->process_.set_preprocessor_config_dir(dir_path);

            // set listen addr
            auto listen_addr = "unix://" + (temp_dir / "listen_addr.sock").string();
            context_->process_.set_preprocessor_listen_addr(listen_addr);

            absl::Status status = context_->process_.launch();
            if (!status.ok())
                co_return std::move(status);
            auto channel = grpc::CreateChannel(listen_addr, grpc::InsecureChannelCredentials());
            context_->stub_ = Predict::NewStub(channel);

            spdlog::info("PyPreprocessingModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         context_->process_.get_preprocessor_config_dir(),
                         fmt::join(input_names(), ", "),
                         fmt::join(output_names(), ", "));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return r;
}

awaitable_result<std::unique_ptr<PyPreprocessingModelOutput>>
PyPreprocessingModel::do_predict(std::unique_ptr<PyPreprocessingModelInput> input) {
    auto output = std::make_unique<PyPreprocessingModelOutput>();
    grpc::ClientContext client_context;
    agrpc::GrpcContext& grpc_context = GrpcClientContextPool::get_instance().get_next();
    grpc::Status status;
    const auto reader = agrpc::request(&Predict::Stub::AsyncPredict, *context_->stub_, client_context, input->request, grpc_context);
    co_await agrpc::finish(reader, output->reply, status, boost::asio::bind_executor(grpc_context, boost::asio::use_awaitable));
    if (!status.ok())
        co_return absl::FailedPreconditionError(fmt::format("preprocessing failed: {}", status.error_message()));
    co_return output;
}

std::string PyPreprocessingModel::info() const {
    return fmt::format("Python preprocessing model loaded from {}",
                       context_->process_.get_preprocessor_config_dir());
}

const std::vector<std::string> &PyPreprocessingModel::input_names() const {
    return context_->process_.get_input_names();
}

const std::vector<std::string> &PyPreprocessingModel::output_names() const {
    return context_->process_.get_output_names();
}

PyPreprocessingModel::~PyPreprocessingModel() {}

} // namespace metaspore::serving
