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

#include <exception>
#include <future>
#include <iostream>
#include <metaspore/actor_process.h>
#include <metaspore/message.h>
#include <metaspore/message_meta.h>
#include <metaspore/ps_runner.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/thread_utils.h>
#include <spdlog/spdlog.h>
#include <thread>

namespace metaspore {

void PSRunner::RunPS(std::shared_ptr<metaspore::ActorConfig> config) {
    // spdlog::set_level(spdlog::level::debug);
    spdlog::info("PS job with coordinator address {}:{} started.", config->GetRootUri(),
                 config->GetRootPort());
    spdlog::info("PSRunner::RunPS: {}", GetThreadIdentifier());
    if (!config->IsLocalMode()) {
        if (config->GetNodeRole() == NodeRole::Coordinator)
            RunPSCoordinator(config);
        else if (config->GetNodeRole() == NodeRole::Server)
            RunPSServer(config);
        else
            RunPSWorker(config);
    } else {
        std::vector<std::future<void>> futures;
        std::packaged_task<void()> coordinator_task([config] { RunPSCoordinator(config); });
        futures.push_back(coordinator_task.get_future());
        std::thread coordinator_thread(std::move(coordinator_task));
        coordinator_thread.detach();
        for (int i = 0; i < config->GetServerCount(); i++) {
            std::packaged_task<void()> server_task([config] { RunPSServer(config); });
            futures.push_back(server_task.get_future());
            std::thread server_thread(std::move(server_task));
            server_thread.detach();
        }
        for (int i = 0; i < config->GetWorkerCount(); i++) {
            std::packaged_task<void()> worker_task([config] { RunPSWorker(config); });
            futures.push_back(worker_task.get_future());
            std::thread worker_thread(std::move(worker_task));
            worker_thread.detach();
        }
        for (size_t i = 0; i < futures.size(); i++)
            futures.at(i).get();
    }
    spdlog::info("PS job with coordinator address {}:{} stopped.", config->GetRootUri(),
                 config->GetRootPort());
}

void PSRunner::RunPSCoordinator(std::shared_ptr<metaspore::ActorConfig> config) {
    spdlog::info("PSRunner::RunPSCoordinator: {}", GetThreadIdentifier());
    try {
        config = config->Copy();
        config->SetNodeRole(NodeRole::Coordinator);
        metaspore::ActorProcess actor(config);
        actor.Run();
    } catch (const std::exception &e) {
        spdlog::error("RunPSCoordinator: {}", e.what());
        throw;
    }
}

void PSRunner::RunPSServer(std::shared_ptr<metaspore::ActorConfig> config) {
    spdlog::info("PSRunner::RunPSServer: {}", GetThreadIdentifier());
    try {
        config = config->Copy();
        config->SetNodeRole(NodeRole::Server);
        metaspore::ActorProcess actor(config);
        actor.Run();
    } catch (const std::exception &e) {
        spdlog::error("RunPSServer: {}", e.what());
        throw;
    }
}

void PSRunner::RunPSWorker(std::shared_ptr<metaspore::ActorConfig> config) {
    spdlog::info("PSRunner::RunPSWorker: {}", GetThreadIdentifier());
    try {
        config = config->Copy();
        config->SetNodeRole(NodeRole::Worker);
        metaspore::ActorProcess actor(config);
        actor.Run();
    } catch (const std::exception &e) {
        spdlog::error("RunPSWorker: {}", e.what());
        throw;
    }
}

} // namespace metaspore
