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

#include <common/arrow/arrow_record_batch_serde.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace metaspore {

result<std::shared_ptr<arrow::RecordBatch>>
ArrowRecordBatchSerde::deserialize_from(const std::string &name, const metaspore::serving::PredictRequest &request) {
    auto find = request.payload().find(name);
    if (find == request.payload().end()) {
        return absl::NotFoundError(fmt::format("Cannot find input {} from request", name));
    }
    const std::string &buffer = find->second;
    auto reader = arrow::Buffer::GetReader(arrow::Buffer::Wrap(buffer.c_str(), buffer.length()));
    if (!reader.ok()) {
        return absl::InternalError(fmt::format("Create reader from buffer failed for input {}: {}",
                                               name, reader.status()));
    }
    auto rb_reader_result = arrow::ipc::RecordBatchFileReader::Open(reader->get());
    if (!rb_reader_result.ok()) {
        return absl::InternalError(
            fmt::format("Create RecordBatchFileReader failed {}", rb_reader_result.status()));
    }
    auto rb_reader = *rb_reader_result;
    if (rb_reader->num_record_batches() <= 0) {
        return absl::NotFoundError("Input record batch is empty");
    }
    auto result = rb_reader->ReadRecordBatch(0);
    if (!result.ok()) {
        return absl::InternalError(fmt::format("Reader record batch failed {}", result.status()));
    }
    return *result;
}

} // namespace metaspore
