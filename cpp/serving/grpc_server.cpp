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
#include <serving/grpc_server.h>
#include <serving/grpc_server_shutdown.h>
#include <serving/metaspore.grpc.pb.h>
#include <serving/model_manager.h>
#include <serving/types.h>
#include <serving/shared_grpc_server_builder.h>
#include <serving/shared_grpc_context.h>
#include <metaspore/string_utils.h>

#include <boost/asio/bind_executor.hpp>
#include <fmt/format.h>
#include <gflags/gflags.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <optional>

namespace metaspore::serving {

DECLARE_string(grpc_listen_host);
DECLARE_string(grpc_listen_port);

class GrpcServerContext {
  public:
    GrpcServerContext()
            : builder(SharedGrpcServerBuilder::get_instance())
            , predict_service()
            , load_service()
            , grpc_context(SharedGrpcContext::get_instance()) {
        spdlog::info("Listening on {}:{}", FLAGS_grpc_listen_host, FLAGS_grpc_listen_port);
        builder->AddListeningPort(
            fmt::format("{}:{}", FLAGS_grpc_listen_host, FLAGS_grpc_listen_port),
            grpc::InsecureServerCredentials());
        builder->RegisterService(&predict_service);
        builder->RegisterService(&load_service);
        server = builder->BuildAndStart();
    }

    std::shared_ptr<grpc::ServerBuilder> builder;
    Predict::AsyncService predict_service;
    Load::AsyncService load_service;
    std::shared_ptr<agrpc::GrpcContext> grpc_context;
    std::unique_ptr<grpc::Server> server;
};

GrpcServer::GrpcServer() { context_ = std::make_unique<GrpcServerContext>(); }

GrpcServer::~GrpcServer() = default;

GrpcServer::GrpcServer(GrpcServer &&) = default;

awaitable<void> respond_error(grpc::ServerAsyncResponseWriter<PredictReply> &writer,
                              const status &s) {
    co_await agrpc::finish_with_error(
        writer, grpc::Status(static_cast<grpc::StatusCode>(s.code()), s.ToString()),
        boost::asio::use_awaitable);
}

awaitable<void> respond_error(grpc::ServerAsyncResponseWriter<LoadReply> &writer,
                              const status &s) {
    co_await agrpc::finish_with_error(
        writer, grpc::Status(static_cast<grpc::StatusCode>(s.code()), s.ToString()),
        boost::asio::use_awaitable);
}

void GrpcServer::run() {
    GrpcServerShutdown server_shutdown{*context_->server, *context_->grpc_context};

    agrpc::repeatedly_request(
        &Predict::AsyncService::RequestPredict, context_->predict_service,
        boost::asio::bind_executor(
            *context_->grpc_context,
            [&](grpc::ServerContext &ctx, PredictRequest &req,
                grpc::ServerAsyncResponseWriter<PredictReply> &writer) -> awaitable<void> {
                auto find_model = ModelManager::get_model_manager().get_model(req.model_name());
                if (!find_model.ok()) {
                    co_await respond_error(writer, find_model.status());
                } else {
                    // convert grpc to fe input
                    std::string ex;
                    try {
                        auto reply_result = co_await(*find_model)->predict(req);
                        if (!reply_result.ok()) {
                            co_await respond_error(writer, reply_result.status());
                        } else {
                            co_await agrpc::finish(writer, *reply_result, grpc::Status::OK,
                                                   boost::asio::use_awaitable);
                        }
                    } catch (const std::exception &e) {
                        // unknown exception
                        ex = e.what();
                    }
                    if (!ex.empty())
                        co_await respond_error(writer, absl::UnknownError(std::move(ex)));
                }
                co_return;
            }));

    agrpc::repeatedly_request(
        &Load::AsyncService::RequestLoad, context_->load_service,
        boost::asio::bind_executor(
            *context_->grpc_context,
            [&](grpc::ServerContext &ctx, LoadRequest &req,
                grpc::ServerAsyncResponseWriter<LoadReply> &writer) -> awaitable<void> {
                const std::string &model_name = req.model_name();
                const std::string &version = req.version();
                const std::string &dir_path = req.dir_path();
                std::string desc = " model " + metaspore::ToSource(model_name) +
                                   " version " + metaspore::ToSource(version) +
                                   " from " + metaspore::ToSource(dir_path) + ".";
                spdlog::info("Loading" + desc);
                auto status = co_await ModelManager::get_model_manager().load(dir_path, model_name);
                if (!status.ok()) {
                    spdlog::error("Fail to load" + desc);
                    co_await respond_error(writer, status);
                } else {
                    LoadReply reply;
                    reply.set_msg("Successfully loaded" + desc);
                    spdlog::info(reply.msg());
                    co_await agrpc::finish(writer, reply, grpc::Status::OK,
                                           boost::asio::use_awaitable);
                }
                co_return;
            }));

    spdlog::info("Start to accept grpc requests");
    context_->grpc_context->run();
}

} // namespace metaspore::serving
