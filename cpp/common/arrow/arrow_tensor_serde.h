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

#include <common/metaspore.pb.h>
#include <common/types.h>

#include <arrow/tensor.h>
#include <fmt/format.h>

namespace metaspore {

class ArrowTensorSerde {
  public:
    static result<std::shared_ptr<arrow::Tensor>>
    deserialize_from(const std::string &name, metaspore::serving::PredictRequest &request);

    static result<std::shared_ptr<arrow::Tensor>>
    deserialize_from(const std::string &name, metaspore::serving::PredictReply &reply);

    static status serialize_to(const std::string &name, const arrow::Tensor &tensor,
                               metaspore::serving::PredictRequest &request);

    static status serialize_to(const std::string &name, const arrow::Tensor &tensor,
                               metaspore::serving::PredictReply &reply);
};

} // namespace metaspore
