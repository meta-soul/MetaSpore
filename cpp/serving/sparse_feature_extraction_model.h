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

namespace metaspore::serving {

class SparseFeatureExtractionModelContext;

class SparseFeatureExtractionModelOutput : public ModelOutputBase {
  public:
    std::shared_ptr<arrow::RecordBatch> values;
};

class SparseFeatureExtractionModel : public ModelBaseCRTP<SparseFeatureExtractionModel> {
  public:
    using InputType = FeatureExtractionModelInput;
    using OutputType = SparseFeatureExtractionModelOutput;

    SparseFeatureExtractionModel();
    SparseFeatureExtractionModel(SparseFeatureExtractionModel &&);

    awaitable_status load(std::string dir_path) override;

    awaitable_result<std::unique_ptr<OutputType>> do_predict(std::unique_ptr<InputType> input);

    std::string info() const override;

    const std::vector<std::string> &input_names() const override;

    const std::vector<std::string> &output_names() const override;

    ~SparseFeatureExtractionModel();

  private:
    std::unique_ptr<SparseFeatureExtractionModelContext> context_;
};

} // namespace metaspore::serving
