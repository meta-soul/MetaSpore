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

#include <serving/converters.h>
#include <serving/grpc_input_output.h>
#include <serving/model_base.h>

namespace metaspore::serving {

class GrpcModelRunner {
  public:
    virtual ~GrpcModelRunner();
    virtual awaitable_result<PredictReply> predict(PredictRequest &request) = 0;

    std::unique_ptr<Converter> input_conveter;
    std::unique_ptr<Converter> output_conveter;
    std::unique_ptr<ModelBase> model;
};

class GrpcTabularModelRunner : public GrpcModelRunner {
  public:
    awaitable_result<PredictReply> predict(PredictRequest &request) override;
};

} // namespace metaspore::serving