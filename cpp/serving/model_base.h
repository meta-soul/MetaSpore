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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <boost/core/demangle.hpp>
#include <fmt/format.h>

#include <common/threadpool.h>
#include <common/types.h>

namespace metaspore::serving {

class ModelInputOutput {
  public:
    virtual ~ModelInputOutput() {}
};

class ModelInputBase : public ModelInputOutput {
  public:
    virtual ~ModelInputBase() {}
};

class ModelOutputBase : public ModelInputOutput {
  public:
    virtual ~ModelOutputBase() {}
};

class ModelBase {
  public:
    virtual ~ModelBase() {}

    virtual awaitable_status load(std::string dir_path) = 0;

    virtual awaitable_result<std::unique_ptr<ModelInputOutput>>
    predict(std::unique_ptr<ModelInputOutput> input) = 0;

    virtual std::string info() const = 0;

    virtual const std::vector<std::string> &input_names() const = 0;
    virtual const std::vector<std::string> &output_names() const = 0;
};

template <typename Model> class ModelBaseCRTP : public ModelBase {
  public:
    awaitable_result<std::unique_ptr<ModelInputOutput>>
    predict(std::unique_ptr<ModelInputOutput> input) override {
        using InputType = typename Model::InputType;
        using OutputType = typename Model::OutputType;

        auto &tp = Threadpools::get_compute_threadpool();
        auto result = co_await boost::asio::co_spawn(
            tp,
            [&input, this]() mutable -> awaitable_result<std::unique_ptr<OutputType>> {
                InputType *model_input = dynamic_cast<InputType *>(input.get());
                if (model_input == nullptr) {
                    auto p = input.get(); // to avoid evaluation in typeid warning
                    co_return absl::InvalidArgumentError(
                        fmt::format("{} input type error, required {} but got {}",
                                    boost::core::demangle(typeid(Model).name()),
                                    boost::core::demangle(typeid(InputType).name()),
                                    boost::core::demangle(typeid(*p).name())));
                }
                auto result = co_await static_cast<Model *>(this)->do_predict(
                    std::unique_ptr<InputType>(static_cast<InputType *>(input.release())));
                co_return result;
            },
            boost::asio::use_awaitable);
        co_return result;
    }
};

} // namespace metaspore::serving
