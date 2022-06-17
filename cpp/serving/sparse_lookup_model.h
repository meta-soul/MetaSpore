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

#include <serving/model_base.h>

#include <arrow/tensor.h>

namespace metaspore::serving {

class SparseLookupModelGlobal;
class SparseLookupModelContext;

class SparseLookupModelInput : public ModelInputBase {
  public:
    std::vector<uint64_t> indices_holder;
    std::vector<uint64_t> offsets_holder;
    std::shared_ptr<arrow::UInt64Tensor> indices;
    std::shared_ptr<arrow::UInt64Tensor> offsets;
    int64_t batch_size;
};

class SparseLookupModelOutput : public ModelOutputBase {
  public:
    std::vector<uint64_t> indices_holder;
    std::vector<uint64_t> offsets_holder;
    // Inplace modified indices by HashUniquifier
    std::shared_ptr<arrow::UInt64Tensor> indices;
    // Lookup values of keys with last one more dimension of emb_size
    std::shared_ptr<arrow::FloatTensor> values;
    // Uniquified keys
    std::vector<uint64_t> keys;
    // Unmodified offsets
    std::shared_ptr<arrow::UInt64Tensor> offsets;
    int64_t batch_size;
};

class SparseLookupModel : public ModelBaseCRTP<SparseLookupModel> {
  public:
    using InputType = SparseLookupModelInput;
    using OutputType = SparseLookupModelOutput;

    class SparseLookupSource {
      public:
        virtual ~SparseLookupSource() {}
        virtual awaitable_status load(const std::string &dir, GrpcClientContextPool &contexts) = 0;

        virtual awaitable_result<std::shared_ptr<arrow::FloatTensor>>
        lookup(std::shared_ptr<arrow::UInt64Tensor> indices) = 0;

        virtual awaitable_result<uint64_t> get_vector_size() = 0;
    };

    SparseLookupModel();
    SparseLookupModel(SparseLookupModel &&);

    awaitable_status load(std::string dir_path, GrpcClientContextPool &contexts) override;

    awaitable_result<std::unique_ptr<SparseLookupModelOutput>>
    do_predict(std::unique_ptr<SparseLookupModelInput> input);

    std::string info() const override;

    const std::vector<std::string> &input_names() const override;
    const std::vector<std::string> &output_names() const override;

    awaitable_result<uint64_t> get_vector_size();

    ~SparseLookupModel();

  private:
    std::unique_ptr<SparseLookupModelContext> context_;
};

} // namespace metaspore::serving
