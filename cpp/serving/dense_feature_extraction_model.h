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

#include <serving/feature_extraction_model_input.h>

#include <arrow/tensor.h>

namespace metaspore::serving {

class DenseFeatureExtractionModelContext;

class DenseFeatureExtractionModelOutput : public ModelOutputBase {
  public:
    std::unordered_map<std::string, std::shared_ptr<arrow::FloatTensor>> feature_tensors;
};

class DenseFeatureExtractionModel : public ModelBaseCRTP<DenseFeatureExtractionModel> {
  public:
    using InputType = FeatureExtractionModelInput;
    using OutputType = DenseFeatureExtractionModelOutput;

    DenseFeatureExtractionModel();
    DenseFeatureExtractionModel(DenseFeatureExtractionModel &&);

    awaitable_status load(std::string dir_path) override;

    awaitable_result<std::unique_ptr<OutputType>> do_predict(std::unique_ptr<InputType> input);

    std::string info() const override;

    const std::vector<std::string> &input_names() const override;

    const std::vector<std::string> &output_names() const override;

    ~DenseFeatureExtractionModel();

  private:
    std::unique_ptr<DenseFeatureExtractionModelContext> context_;
};

} // namespace metaspore::serving