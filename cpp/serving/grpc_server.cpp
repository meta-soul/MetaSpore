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
#include <serving/metaspore.grpc.pb.h>
#include <serving/model_manager.h>
#include <serving/types.h>

#include <agrpc/asioGrpc.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/asio/signal_set.hpp>
#include <fmt/format.h>
#include <gflags/gflags.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <optional>

namespace metaspore::serving {

DECLARE_string(grpc_listen_host);
DECLARE_string(grpc_listen_port);

// From
// https://github.com/Tradias/asio-grpc/blob/cd9d3aad3af56a43793594d1a5388a453e7d5668/example/streaming-server.cpp#L31
// Copyright 2022 Dennis Hezel
struct ServerShutdown {
    grpc::Server &server;
    boost::asio::basic_signal_set<agrpc::GrpcContext::executor_type> signals;
    std::optional<std::thread> shutdown_thread;

    ServerShutdown(grpc::Server &server, agrpc::GrpcContext &grpc_context)
        : server(server), signals(grpc_context, SIGINT, SIGTERM) {
        signals.async_wait([&](auto &&, auto &&signal) {
            spdlog::info("Shutdown with signal {}", signal);
            shutdown();
        });
    }

    void shutdown() {
        if (!shutdown_thread) {
            // This will cause all coroutines to run to completion normally
            // while returning `false` from RPC related steps, cancelling the signal
            // so that the GrpcContext will eventually run out of work and return
            // from `run()`.
            shutdown_thread.emplace([&] {
                signals.cancel();
                server.Shutdown();
            });
            // Alternatively call `grpc_context.stop()` here instead which causes all coroutines
            // to end at their next suspension point.
            // Then call `server->Shutdown()` after the call to `grpc_context.run()` returns
            // or `.reset()` the grpc_context and go into another `grpc_context.run()`
        }
    }

    ~ServerShutdown() {
        if (shutdown_thread) {
            shutdown_thread->join();
        }
    }
};

class GrpcServerContext {
  public:
    GrpcServerContext() : builder(), service(), grpc_context(builder.AddCompletionQueue()) {
        spdlog::info("Listening on {}:{}", FLAGS_grpc_listen_host, FLAGS_grpc_listen_port);
        builder.AddListeningPort(
            fmt::format("{}:{}", FLAGS_grpc_listen_host, FLAGS_grpc_listen_port),
            grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
    }

    grpc::ServerBuilder builder;
    Predict::AsyncService service;
    agrpc::GrpcContext grpc_context;
    std::unique_ptr<grpc::Server> server;
};

GrpcServer::GrpcServer() { context_ = std::make_unique<GrpcServerContext>(); }

GrpcServer::~GrpcServer() = default;

GrpcServer::GrpcServer(GrpcServer &&) = default;

// From
// https://github.com/Tradias/asio-grpc/blob/f179621e3ff5401b99e4c40ba2427a1a1ab7ffcf/example/helper/coSpawner.hpp
// Copyright 2022 Dennis Hezel
template <class Handler> struct CoSpawner {
    using executor_type = boost::asio::associated_executor_t<Handler>;
    using allocator_type = boost::asio::associated_allocator_t<Handler>;

    Handler handler;

    explicit CoSpawner(Handler handler) : handler(std::move(handler)) {}

    template <class T> void operator()(agrpc::RepeatedlyRequestContext<T> &&request_context) {
        boost::asio::co_spawn(
            this->get_executor(),
            [handler = std::move(handler), request_context = std::move(request_context)]() mutable
            -> boost::asio::awaitable<void> {
                co_await std::apply(std::move(handler), request_context.args());
            },
            boost::asio::detached);
    }

    [[nodiscard]] executor_type get_executor() const noexcept {
        return boost::asio::get_associated_executor(handler);
    }

    [[nodiscard]] allocator_type get_allocator() const noexcept {
        return boost::asio::get_associated_allocator(handler);
    }
};

awaitable<void> respond_error(grpc::ServerAsyncResponseWriter<PredictReply> &writer,
                              const status &s) {
    co_await agrpc::finish_with_error(
        writer, grpc::Status(static_cast<grpc::StatusCode>(s.code()), s.ToString()),
        boost::asio::use_awaitable);
}

void GrpcServer::run() {
    ServerShutdown server_shutdown{*context_->server, context_->grpc_context};

    agrpc::repeatedly_request(
        &Predict::AsyncService::RequestPredict, context_->service,
        CoSpawner{boost::asio::bind_executor(
            context_->grpc_context,
            [&](grpc::ServerContext &ctx, PredictRequest &req,
                grpc::ServerAsyncResponseWriter<PredictReply> writer) -> awaitable<void> {
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
            })});

    spdlog::info("Start to accept grpc requests");
    context_->grpc_context.run();
}

} // namespace metaspore::serving