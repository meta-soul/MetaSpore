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

#include <onnxruntime_cxx_api.h>
#include <serving/model_base.h>
#include <unordered_map>

namespace metaspore::serving {

class OrtModelGlobal;
class OrtModelContext;

class OrtModelInput : public ModelInputBase {
  public:
    struct Value {
        // a possible memory holder, could be null
        std::unique_ptr<ModelInputOutput> holder;
        Ort::Value value;
    };
    std::unordered_map<std::string, Value> inputs;
};

class OrtModelOutput : public ModelOutputBase {
  public:
    std::unordered_map<std::string, Ort::Value> outputs;
};

class OrtModel : public ModelBaseCRTP<OrtModel> {
  public:
    using InputType = OrtModelInput;
    using OutputType = OrtModelOutput;

    OrtModel();
    OrtModel(OrtModel &&);

    awaitable_status load(std::string dir_path, GrpcClientContextPool &contexts) override;

    awaitable_result<std::unique_ptr<OutputType>> do_predict(std::unique_ptr<InputType> input);

    std::string info() const override;

    const std::vector<std::string> &input_names() const override;
    const std::vector<std::string> &output_names() const override;

    ~OrtModel();

  private:
    std::unique_ptr<OrtModelContext> context_;
};

} // namespace metaspore::serving
