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

#include <serving/arrow_tensor_serde.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace metaspore::serving {

result<std::shared_ptr<arrow::Tensor>>
ArrowTensorSerde::deserialize_from(const std::string &name, PredictRequest &request) {
    auto find = request.payload().find(name);
    if (find == request.payload().end()) {
        return absl::NotFoundError(fmt::format("Cannot find input {} from request", name));
    }
    const std::string &buffer = find->second;
    arrow::io::BufferReader reader((const uint8_t *) buffer.data(), (int64_t) buffer.size());
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto tensor, arrow::ipc::ReadTensor(&reader));
    return tensor;
}

result<std::shared_ptr<arrow::Tensor>>
ArrowTensorSerde::deserialize_from(const std::string &name, PredictReply &reply) {
    auto find = reply.payload().find(name);
    if (find == reply.payload().end()) {
        return absl::NotFoundError(fmt::format("Cannot find input {} from reply", name));
    }
    const std::string &buffer = find->second;
    arrow::io::BufferReader reader((const uint8_t *) buffer.data(), (int64_t) buffer.size());
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto tensor, arrow::ipc::ReadTensor(&reader));
    return tensor;
}

status ArrowTensorSerde::serialize_to(const std::string &name, const arrow::Tensor &tensor,
                                      PredictRequest &request) {
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto buffer_stream_result,
                                   arrow::io::BufferOutputStream::Create());
    int32_t meta_len = 0;
    int64_t length = 0L;
    CALL_AND_RETURN_IF_STATUS_NOT_OK(
        arrow::ipc::WriteTensor(tensor, buffer_stream_result.get(), &meta_len, &length));
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto buffer_result, buffer_stream_result->Finish());
    (*request.mutable_payload())[name] = buffer_result->ToString();
    return absl::OkStatus();
}

status ArrowTensorSerde::serialize_to(const std::string &name, const arrow::Tensor &tensor,
                                      PredictReply &reply) {
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto buffer_stream_result,
                                   arrow::io::BufferOutputStream::Create());
    int32_t meta_len = 0;
    int64_t length = 0L;
    CALL_AND_RETURN_IF_STATUS_NOT_OK(
        arrow::ipc::WriteTensor(tensor, buffer_stream_result.get(), &meta_len, &length));
    ASSIGN_RESULT_OR_RETURN_NOT_OK(auto buffer_result, buffer_stream_result->Finish());
    (*reply.mutable_payload())[name] = buffer_result->ToString();
    return absl::OkStatus();
}

} // namespace metaspore::serving
