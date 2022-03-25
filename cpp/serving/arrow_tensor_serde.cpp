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