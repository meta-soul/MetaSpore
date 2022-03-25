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

#include <serving/feature_extraction_model_input.h>
#include <serving/grpc_model_runner.h>

namespace metaspore::serving {

GrpcModelRunner::~GrpcModelRunner() = default;

awaitable_result<PredictReply> GrpcTabularModelRunner::predict(PredictRequest &request) {
    // convert grpc request to sparse fe input
    auto req = std::make_unique<GrpcRequestOutput>(request);
    auto input = std::make_unique<FeatureExtractionModelInput>();
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(input_conveter->convert_input(std::move(req), input.get()));

    // do prediction
    CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto predict_result, model->predict(std::move(input)));

    // convert prediction output to grpc reply
    PredictReply reply;
    auto reply_ptr = std::make_unique<GrpcReplyInput>(reply);
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
        output_conveter->convert_input(std::move(predict_result), reply_ptr.get()));

    co_return reply;
}

} // namespace metaspore::serving