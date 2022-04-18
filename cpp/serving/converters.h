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

#include <string>
#include <vector>

#include <serving/model_base.h>
#include <serving/types.h>

#include <boost/core/demangle.hpp>
#include <fmt/format.h>

namespace Ort {
class Value;
}

namespace arrow {
class Tensor;
}

namespace metaspore::serving {

class Converter {
  public:
    Converter() {}
    Converter(std::string name) : names_({std::move(name)}) {}
    Converter(std::vector<std::string> names) : names_(std::move(names)) {}

    virtual status convert_input(std::unique_ptr<ModelInputOutput> from, ModelInputOutput *to) = 0;

    virtual ~Converter() {}

    static result<std::shared_ptr<arrow::Tensor>> ort_to_arrow_tensor(const Ort::Value &ort_tensor);

    template <typename T> static Ort::Value arrow_to_ort_tensor(const arrow::Tensor &arrow_tensor);

  protected:
    std::vector<std::string> names_;
};

template <typename Conv> class ConverterCRTP : public Converter {
  public:
    using Converter::Converter;
    status convert_input(std::unique_ptr<ModelInputOutput> from, ModelInputOutput *to) override {
        if (!from) {
            return absl::InvalidArgumentError(fmt::format("Cannot convert null pointer"));
        }
        using IN = typename Conv::InputType;
        using OUT = typename Conv::OutputType;
        ModelInputOutput *from_p = from.release();
        auto &&type_from = typeid(*from_p);
        if (type_from != typeid(IN)) {
            return absl::InvalidArgumentError(fmt::format(
                "Converter {} requires {} as input type, but got {}",
                boost::core::demangle(typeid(Conv).name()),
                boost::core::demangle(typeid(IN).name()), boost::core::demangle(type_from.name())));
        }
        auto &&type_to = typeid(*to);
        if (type_to != typeid(OUT)) {
            return absl::InvalidArgumentError(fmt::format(
                "Converter {} requires {} as output type, but got {}",
                boost::core::demangle(typeid(Conv).name()),
                boost::core::demangle(typeid(OUT).name()), boost::core::demangle(type_to.name())));
        }
        return static_cast<Conv *>(this)->convert(std::unique_ptr<IN>(static_cast<IN *>(from_p)),
                                                  static_cast<OUT *>(to));
    }
};

class GrpcRequestOutput;
class FeatureExtractionModelInput;
class OrtModelInput;

class GrpcRequestToFEConverter : public ConverterCRTP<GrpcRequestToFEConverter> {
  public:
    using InputType = GrpcRequestOutput;
    using OutputType = FeatureExtractionModelInput;
    using ConverterCRTP<GrpcRequestToFEConverter>::ConverterCRTP;

    status convert(std::unique_ptr<GrpcRequestOutput> from, FeatureExtractionModelInput *to);
};

class GrpcRequestToOrtConverter : public ConverterCRTP<GrpcRequestToOrtConverter> {
  public:
    using InputType = GrpcRequestOutput;
    using OutputType = OrtModelInput;
    using ConverterCRTP<GrpcRequestToOrtConverter>::ConverterCRTP;

    status convert(std::unique_ptr<GrpcRequestOutput> from, OrtModelInput *to);
};

class SparseFeatureExtractionModelOutput;
class SparseLookupModelInput;

class SparseFEToLookupConverter : public ConverterCRTP<SparseFEToLookupConverter> {
  public:
    using InputType = SparseFeatureExtractionModelOutput;
    using OutputType = SparseLookupModelInput;
    using ConverterCRTP<SparseFEToLookupConverter>::ConverterCRTP;

    status convert(std::unique_ptr<SparseFeatureExtractionModelOutput> from,
                   SparseLookupModelInput *to);
};

class DenseFeatureExtractionModelOutput;

class DenseFEToOrtConverter : public ConverterCRTP<DenseFEToOrtConverter> {
  public:
    using InputType = DenseFeatureExtractionModelOutput;
    using OutputType = OrtModelInput;
    using ConverterCRTP<DenseFEToOrtConverter>::ConverterCRTP;

    status convert(std::unique_ptr<DenseFeatureExtractionModelOutput> from, OrtModelInput *to);
};

class OrtModelOutput;
class GrpcReplyInput;

class OrtToGrpcReplyConverter : public ConverterCRTP<OrtToGrpcReplyConverter> {
  public:
    using InputType = OrtModelOutput;
    using OutputType = GrpcReplyInput;
    using ConverterCRTP<OrtToGrpcReplyConverter>::ConverterCRTP;

    status convert(std::unique_ptr<OrtModelOutput> from, GrpcReplyInput *to);
};

} // namespace metaspore::serving