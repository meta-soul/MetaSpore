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

#include <serving/metaspore.pb.h>
#include <serving/types.h>

#include <arrow/record_batch.h>
#include <fmt/format.h>

namespace metaspore::serving {

class ArrowRecordBatchSerde {
  public:
    static result<std::shared_ptr<arrow::RecordBatch>>
    deserialize_from(const std::string &name, const PredictRequest &request);
};

} // namespace metaspore::serving
