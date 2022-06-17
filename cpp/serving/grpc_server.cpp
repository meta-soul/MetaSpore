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

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <optional>
#include <forward_list>

namespace metaspore::serving {

DECLARE_string(grpc_listen_host);
DECLARE_string(grpc_listen_port);
DECLARE_uint64(grpc_server_threads);

class GrpcServerContext {
  public:
    GrpcServerContext(GrpcClientContextPool &client_context_pool)
        : client_context_pool(client_context_pool) {
        grpc_server_thread_count = FLAGS_grpc_server_threads;
        if (grpc_server_thread_count == 0)
            grpc_server_thread_count = std::thread::hardware_concurrency();
        grpc::ServerBuilder builder;
        for (int i = 0; i < grpc_server_thread_count; i++)
            grpc_server_contexts.emplace_front(builder.AddCompletionQueue());
        spdlog::info("Listening on {}:{}", FLAGS_grpc_listen_host, FLAGS_grpc_listen_port);
        builder.AddListeningPort(
            fmt::format("{}:{}", FLAGS_grpc_listen_host, FLAGS_grpc_listen_port),
            grpc::InsecureServerCredentials());
        builder.RegisterService(&predict_service);
        builder.RegisterService(&load_service);
        server = builder.BuildAndStart();
    }

    std::unique_ptr<grpc::Server> server;
    Predict::AsyncService predict_service;
    Load::AsyncService load_service;
    std::forward_list<agrpc::GrpcContext> grpc_server_contexts;
    std::vector<std::thread> grpc_server_threads;
    int grpc_server_thread_count{};
    GrpcClientContextPool &client_context_pool;
};

GrpcServer::GrpcServer(GrpcClientContextPool &client_context_pool)
{
    context_ = std::make_unique<GrpcServerContext>(client_context_pool);
}

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

void register_predict_request_handler(agrpc::GrpcContext &grpc_context,
                                      Predict::AsyncService &predict_service)
{
    agrpc::repeatedly_request(
        &Predict::AsyncService::RequestPredict, predict_service,
        boost::asio::bind_executor(
            grpc_context,
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
}

void register_load_request_handler(agrpc::GrpcContext &grpc_context,
                                   Load::AsyncService &load_service,
                                   GrpcClientContextPool &client_context_pool)
{
    agrpc::repeatedly_request(
        &Load::AsyncService::RequestLoad, load_service,
        boost::asio::bind_executor(
            grpc_context,
            [&](grpc::ServerContext &ctx, LoadRequest &req,
                grpc::ServerAsyncResponseWriter<LoadReply> &writer) -> awaitable<void> {
                const std::string &model_name = req.model_name();
                const std::string &version = req.version();
                const std::string &dir_path = req.dir_path();
                std::string desc = " model " + metaspore::ToSource(model_name) +
                                   " version " + metaspore::ToSource(version) +
                                   " from " + metaspore::ToSource(dir_path) + ".";
                spdlog::info("Loading" + desc);
                auto status = co_await ModelManager::get_model_manager().load(dir_path, model_name, client_context_pool);
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
}

void GrpcServer::run() {
    GrpcServerShutdown server_shutdown{*context_->server, context_->grpc_server_contexts.front()};
    for (int i = 0; i < context_->grpc_server_thread_count; i++) {
        context_->grpc_server_threads.emplace_back([&, i] {
            auto &grpc_context = *std::next(context_->grpc_server_contexts.begin(), i);
            register_predict_request_handler(grpc_context, context_->predict_service);
            register_load_request_handler(grpc_context, context_->load_service, context_->client_context_pool);
            grpc_context.run();
        });
    }
    spdlog::info("Start to accept grpc requests with {} threads", context_->grpc_server_thread_count);
    for (auto &thread : context_->grpc_server_threads)
        thread.join();
}

} // namespace metaspore::serving
