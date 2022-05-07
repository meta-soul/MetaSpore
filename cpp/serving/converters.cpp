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

#include <common/logger.h>
#include <serving/arrow_helpers.h>
#include <serving/arrow_record_batch_serde.h>
#include <serving/arrow_tensor_serde.h>
#include <serving/converters.h>
#include <serving/dense_feature_extraction_model.h>
#include <serving/grpc_input_output.h>
#include <serving/py_preprocessing_model.h>
#include <serving/ort_model.h>
#include <serving/sparse_feature_extraction_model.h>
#include <serving/sparse_lookup_model.h>
#include <serving/utils.h>

namespace metaspore::serving {

template <typename T> Ort::Value Converter::arrow_to_ort_tensor(const arrow::Tensor &arrow_tensor) {
    const auto &shape = arrow_tensor.shape();
    const T *data = (const T *)arrow_tensor.raw_data();
    Ort::Value ort_tensor = Ort::Value::CreateTensor<T>(
        Ort::AllocatorWithDefaultOptions().GetInfo(), const_cast<T *>(data),
        (size_t)arrow_tensor.size(), shape.data(), shape.size());
    return ort_tensor;
}

template Ort::Value Converter::arrow_to_ort_tensor<float>(const arrow::Tensor &arrow_tensor);
template Ort::Value Converter::arrow_to_ort_tensor<int32_t>(const arrow::Tensor &arrow_tensor);
template Ort::Value Converter::arrow_to_ort_tensor<int64_t>(const arrow::Tensor &arrow_tensor);
template Ort::Value Converter::arrow_to_ort_tensor<double>(const arrow::Tensor &arrow_tensor);

// specialized with no static type
template <> Ort::Value Converter::arrow_to_ort_tensor<void>(const arrow::Tensor &arrow_tensor) {
    auto type = arrow_tensor.type();
    switch (type->id()) {
    case arrow::Type::DOUBLE:
        return arrow_to_ort_tensor<double>(arrow_tensor);
    case arrow::Type::FLOAT:
        return arrow_to_ort_tensor<float>(arrow_tensor);
    case arrow::Type::INT32:
        return arrow_to_ort_tensor<int32_t>(arrow_tensor);
    case arrow::Type::INT64:
        return arrow_to_ort_tensor<int64_t>(arrow_tensor);
    default:
        throw std::runtime_error(fmt::format(
            "Cannot convert unsupported arrow tensor with type to onnx", type->ToString()));
    }
}

status GrpcRequestToFEConverter::convert(std::unique_ptr<GrpcRequestOutput> from,
                                         FeatureExtractionModelInput *to) {
    for (const auto &name : names_) {
        ASSIGN_RESULT_OR_RETURN_NOT_OK(
            auto r, ArrowRecordBatchSerde::deserialize_from(name, from->request));
        to->feature_tables.emplace(name, r);
    }
    return absl::OkStatus();
}

status GrpcRequestToOrtConverter::convert(std::unique_ptr<GrpcRequestOutput> from,
                                          OrtModelInput *to) {
    for (const auto &name : names_) {
        ASSIGN_RESULT_OR_RETURN_NOT_OK(auto r,
                                       ArrowTensorSerde::deserialize_from(name, from->request));
        to->inputs.emplace(name, OrtModelInput::Value{.value = arrow_to_ort_tensor<void>(*r)});
    }
    // move from to the first element to hold all memories
    if (!to->inputs.empty())
        to->inputs.begin()->second.holder = std::move(from);
    return absl::OkStatus();
}

template <typename T> using Container = std::vector<T>;

status SparseFEToLookupConverter::convert(std::unique_ptr<SparseFeatureExtractionModelOutput> from,
                                          SparseLookupModelInput *to) {
    // convert arrow::RecordBatch of uint64 indices to flattened 1d UInt64Tensor and offsets tensor
    // first create std::vectors as value holders
    int64_t rows = from->values->num_rows();
    int64_t cols = from->values->num_columns();
    to->batch_size = rows;
    to->indices_holder.reserve(rows * cols);
    to->offsets_holder.reserve(rows * cols + 1);
    ASSIGN_RESULT_OR_RETURN_NOT_OK(
        auto accessor_make_result,
        HashListAccessor::create_accessor_makers<Container>(from->values->columns()));
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            auto accessor = accessor_make_result[j](i);
            to->offsets_holder.push_back(to->indices_holder.size());
            if (accessor.empty()) {
                // handle null fields
                to->indices_holder.push_back(0UL);
            } else {
                for (auto h : accessor) {
                    to->indices_holder.push_back(h);
                }
            }
        }
    }
    // make indices and offsets tensor from holders
    ASSIGN_RESULT_OR_RETURN_NOT_OK(
        to->indices, arrow::UInt64Tensor::Make(arrow::Buffer::Wrap(to->indices_holder),
                                               {(int64_t)to->indices_holder.size()}));
    ASSIGN_RESULT_OR_RETURN_NOT_OK(
        to->offsets, arrow::UInt64Tensor::Make(arrow::Buffer::Wrap(to->offsets_holder),
                                               {(int64_t)to->offsets_holder.size()}));
    return absl::OkStatus();
}

status DenseFEToOrtConverter::convert(std::unique_ptr<DenseFeatureExtractionModelOutput> from,
                                      OrtModelInput *to) {
    // convert dense fe output (float tensors) to ort values
    for (const auto &[name, arrow_tensor] : from->feature_tensors) {
        Ort::Value ort_tensor = arrow_to_ort_tensor<float>(*arrow_tensor);
        to->inputs.emplace(name, OrtModelInput::Value{.value = std::move(ort_tensor)});
    }
    // move from to the first element to hold all memories
    if (!from->feature_tensors.empty())
        to->inputs.find(from->feature_tensors.begin()->first)->second.holder = std::move(from);
    return absl::OkStatus();
}

result<std::shared_ptr<arrow::Tensor>>
Converter::ort_to_arrow_tensor(const Ort::Value &ort_tensor) {
    // convert ort tensor to (non owning) arrow tensor
    if (!ort_tensor.IsTensor()) {
        return absl::InternalError("Cannot support non tensor ort value");
    }
    Ort::TensorTypeAndShapeInfo ort_type = ort_tensor.GetTensorTypeAndShapeInfo();
    auto arrow_tensor_result = [&]() -> arrow::Result<std::shared_ptr<arrow::Tensor>> {
        switch (ort_type.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return arrow::FloatTensor::Make(
                arrow::Buffer::Wrap(ort_tensor.GetTensorData<float>(), ort_type.GetElementCount()),
                ort_type.GetShape());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return arrow::DoubleTensor::Make(
                arrow::Buffer::Wrap(ort_tensor.GetTensorData<double>(), ort_type.GetElementCount()),
                ort_type.GetShape());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return arrow::Int32Tensor::Make(arrow::Buffer::Wrap(ort_tensor.GetTensorData<int32_t>(),
                                                                ort_type.GetElementCount()),
                                            ort_type.GetShape());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return arrow::Int64Tensor::Make(arrow::Buffer::Wrap(ort_tensor.GetTensorData<int64_t>(),
                                                                ort_type.GetElementCount()),
                                            ort_type.GetShape());
        default:
            return std::shared_ptr<arrow::Tensor>(nullptr);
        }
    }();
    if (!arrow_tensor_result.ok()) {
        return absl::InternalError(
            fmt::format("Unsupported ort type {}", arrow_tensor_result.status()));
    }
    return *arrow_tensor_result;
}

status OrtToGrpcReplyConverter::convert(std::unique_ptr<OrtModelOutput> from, GrpcReplyInput *to) {
    // serialize ort tensor to grpc bytes
    // first convert ort tensor to arrow tensor then serialize it
    for (const auto &name : names_) {
        auto find = from->outputs.find(name);
        if (find == from->outputs.end()) {
            return absl::NotFoundError(
                fmt::format("Cannot find {} tensor from ort model output", name));
        }
        ASSIGN_RESULT_OR_RETURN_NOT_OK(auto arrow_tensor_result,
                                       Converter::ort_to_arrow_tensor(find->second));
        CALL_AND_RETURN_IF_STATUS_NOT_OK(
            ArrowTensorSerde::serialize_to(name, *arrow_tensor_result, to->reply));
    }
    return absl::OkStatus();
}

status GrpcRequestToPyPreprocessingConverter::convert(std::unique_ptr<GrpcRequestOutput> from, PyPreprocessingModelInput *to) {
    *to->request.mutable_payload() = std::move(*from->request.mutable_payload());
    return absl::OkStatus();
}

status PyPreprocessingToOrtConverter::convert(std::unique_ptr<PyPreprocessingModelOutput> from, OrtModelInput *to) {
    for (const auto &name : names_) {
        ASSIGN_RESULT_OR_RETURN_NOT_OK(auto r,
                                       ArrowTensorSerde::deserialize_from(name, from->reply));
        to->inputs.emplace(name, OrtModelInput::Value{.value = arrow_to_ort_tensor<void>(*r)});
    }
    // move from to the first element to hold all memories
    if (!to->inputs.empty())
        to->inputs.begin()->second.holder = std::move(from);
    return absl::OkStatus();
}

} // namespace metaspore::serving
