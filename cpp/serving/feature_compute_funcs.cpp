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

#include <common/hash_utils.h>
#include <serving/arrow_helpers.h>
#include <serving/feature_compute_funcs.h>
#include <common/utils.h>

#include <fmt/format.h>

namespace cp = arrow::compute;

namespace metaspore::serving {

// bkdr hash needs a state to store hash value of "feature_name="
struct StringBKDRHashState : public cp::KernelState {
    uint64_t seed = 0;
};

arrow::Status StringBKDRHashKernelString(cp::KernelContext *ctx, const cp::ExecBatch &batch,
                                         arrow::Datum *out) {
    const StringBKDRHashState *state = (const StringBKDRHashState *)ctx->state();
    auto input_array = batch[0].array_as<arrow::StringArray>();
    arrow::UInt64Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(input_array->length()));
    for (auto &&elem : *input_array) {
        if (elem.has_value()) {
            if (elem->empty()) {
                // empty string is treated as null
                ARROW_RETURN_NOT_OK(builder.AppendNull());
            } else {
                uint64_t hash = BKDRHash(elem->data(), elem->length(), 0);
                ARROW_RETURN_NOT_OK(builder.Append(BKDRHashOneField(state->seed, hash)));
            }
        } else {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
        }
    }
    ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
    *out->mutable_array() = *array->data();
    return arrow::Status::OK();
}

arrow::Status StringBKDRHashKernelListString(cp::KernelContext *ctx, const cp::ExecBatch &batch,
                                             arrow::Datum *out) {
    const StringBKDRHashState *state = (const StringBKDRHashState *)ctx->state();
    auto input_array = batch[0].array_as<arrow::ListArray>();
    auto value_builder = std::make_shared<arrow::UInt64Builder>();
    arrow::ListBuilder builder(ctx->memory_pool(), value_builder,
                               std::make_shared<arrow::ListType>(arrow::uint64()));
    std::shared_ptr<arrow::StringArray> values =
        std::dynamic_pointer_cast<arrow::StringArray>(input_array->values());
    if (!values) {
        return arrow::Status::Invalid(
            "StringBKDRHashKernelListString: input is not a string list array");
    }
    ARROW_RETURN_NOT_OK(value_builder->Reserve(values->length()));
    std::shared_ptr<arrow::Int32Array> offsets_array =
        std::dynamic_pointer_cast<arrow::Int32Array>(input_array->offsets());
    if (!offsets_array) {
        return arrow::Status::Invalid(
            "StringBKDRHashKernelListString: input's offset is not an int32 array");
    }
    ARROW_RETURN_NOT_OK(builder.Reserve(offsets_array->length()));
    const int32_t *offsets = offsets_array->raw_values();
    for (size_t i = 0; i < offsets_array->length() - 1; ++i) {
        if (input_array->IsNull(i)) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
        } else {
            int32_t begin = offsets[i];
            int32_t end = offsets[i + 1];
            if (begin == end) {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
            } else {
                ARROW_RETURN_NOT_OK(builder.Append());
                for (int32_t j = begin; j < end; ++j) {
                    // ignore nulls in each list
                    if (!values->IsNull(j)) {
                        auto elem = values->GetView(j);
                        if (elem.empty()) {
                            ARROW_RETURN_NOT_OK(value_builder->AppendNull());
                        } else {
                            uint64_t hash = BKDRHash(elem.data(), elem.length(), 0);
                            ARROW_RETURN_NOT_OK(
                                value_builder->Append(BKDRHashOneField(state->seed, hash)));
                        }
                    }
                }
            }
        }
    }
    ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
    *out->mutable_array() = *array->data();
    return arrow::Status::OK();
}

static const cp::FunctionDoc bkdr_func_doc{"Perform bkdr hash on string array or string list array",
                                           "Input should be a column of string or list<string>",
                                           {"input"},
                                           "StringBKDRHashFunctionOption"};

arrow::Status AddStringBKDRHashFunction() {
    cp::FunctionRegistry *registry = cp::GetFunctionRegistry();
    auto initfn =
        [](cp::KernelContext *context,
           const cp::KernelInitArgs &args) -> arrow::Result<std::unique_ptr<cp::KernelState>> {
        const StringBKDRHashFunctionOption *option =
            dynamic_cast<const StringBKDRHashFunctionOption *>(args.options);
        if (option == nullptr) {
            return arrow::Status::Invalid("Attempted to call a StringBKDRHash function without "
                                          "StringBKDRHashFunctionOptions");
        }
        const std::string &name = option->name;
        auto state = std::make_unique<StringBKDRHashState>();
        state->seed = BKDRHashWithEqualPostfix(name.c_str(), name.length(), 0);
        return state;
    };
    cp::ScalarKernel string_kernel({cp::InputType::Array(arrow::utf8())}, arrow::uint64(),
                                   /* exec = */ StringBKDRHashKernelString,
                                   /* init = */ initfn);
    string_kernel.can_write_into_slices = false;
    cp::ScalarKernel string_list_kernel(
        {cp::InputType::Array(std::make_shared<arrow::ListType>(arrow::utf8()))},
        std::static_pointer_cast<arrow::DataType>(
            std::make_shared<arrow::ListType>(arrow::uint64())),
        /* exec = */ StringBKDRHashKernelListString,
        /* init = */ initfn);
    string_list_kernel.can_write_into_slices = false;
    auto func =
        std::make_shared<cp::ScalarFunction>("bkdr_hash", cp::Arity::Unary(), &bkdr_func_doc);
    ARROW_RETURN_NOT_OK(func->AddKernel(std::move(string_kernel)));
    ARROW_RETURN_NOT_OK(func->AddKernel(std::move(string_list_kernel)));
    ARROW_RETURN_NOT_OK(registry->AddFunction(func));
    return arrow::Status::OK();
}

static const cp::FunctionDoc bkdr_hash_combine_func_doc{
    "Perform hash combine on multiple list<uint64> array contains string's bkdr hash",
    "Input should be two or more list<uint64> array with equal lengths",
    {"input0"},
    "BKDRHashCombineFunctionOption"};

template <typename T> using Container = std::vector<T>;

arrow::Status BKDRHashCombineKernelListUInt64(cp::KernelContext *ctx, const cp::ExecBatch &batch,
                                              arrow::Datum *out) {
    Container<std::shared_ptr<arrow::Array>> arrays;
    for (const auto &v : batch.values) {
        if (!v.is_array()) {
            return arrow::Status::Invalid("BKDRHashCombineKernelListUInt64 only handles array");
        }
        auto array = v.make_array();
        arrays.push_back(array);
    }

    ARROW_ASSIGN_OR_RAISE(auto accessor_maker,
                          HashListAccessor::create_accessor_makers<Container>(arrays));

    auto value_builder = std::make_shared<arrow::UInt64Builder>();
    arrow::ListBuilder builder(ctx->memory_pool(), value_builder,
                               std::make_shared<arrow::ListType>(arrow::uint64()));
    ARROW_RETURN_NOT_OK(builder.Reserve(batch.length));
    for (int64_t i = 0; i < batch.length; ++i) {
        Container<HashListAccessor> lists;
        lists.reserve(batch.values.size());
        for (size_t j = 0; j < batch.values.size(); ++j) {
            lists.emplace_back(accessor_maker[j](i));
        }
        size_t total_results =
            std::accumulate(lists.begin(), lists.end(), size_t(1),
                            [](size_t mul, auto &&accessor) { return accessor.size() * mul; });
        // lists is empty or one of the list is empty
        if (total_results == 0) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
        } else {
            ARROW_RETURN_NOT_OK(builder.Append());
            auto append = [&](uint64_t hash) { (void)value_builder->Append(hash); };
            CartesianHashCombine<HashListAccessor, Container, decltype(append)>::CombineOneFeature(
                lists, std::move(append), total_results);
        }
    }
    ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
    *out->mutable_array() = *array->data();
    return arrow::Status::OK();
}

arrow::Status AddBKDRHashCombineFunction() {
    cp::FunctionRegistry *registry = cp::GetFunctionRegistry();
    cp::ScalarKernel kernel(
        cp::KernelSignature::Make({cp::InputType(/* ANY_TYPE */ arrow::ValueDescr::ARRAY)},
                                  std::static_pointer_cast<arrow::DataType>(
                                      std::make_shared<arrow::ListType>(arrow::uint64())),
                                  /* is_varargs = */ true),
        /* exec = */ BKDRHashCombineKernelListUInt64);
    kernel.can_write_into_slices = false;
    auto func = std::make_shared<cp::ScalarFunction>("bkdr_hash_combine", cp::Arity::VarArgs(1),
                                                     &bkdr_hash_combine_func_doc);
    ARROW_RETURN_NOT_OK(func->AddKernel(std::move(kernel)));
    ARROW_RETURN_NOT_OK(registry->AddFunction(func));
    return arrow::Status::OK();
}

StringBKDRHashFunctionOption::StringBKDRHashFunctionOption()
    : arrow::compute::FunctionOptions(
          MetaSporeBKDRHashFuncOptType<StringBKDRHashFunctionOption>::get()) {}

StringBKDRHashFunctionOption::StringBKDRHashFunctionOption(const std::string &_name)
    : arrow::compute::FunctionOptions(
          MetaSporeBKDRHashFuncOptType<StringBKDRHashFunctionOption>::get()),
      name(_name) {}

BKDRHashCombineFunctionOption::BKDRHashCombineFunctionOption()
    : arrow::compute::FunctionOptions(
          MetaSporeBKDRHashFuncOptType<BKDRHashCombineFunctionOption>::get()) {}

arrow::Status RegisterAllFunctions() {
    ARROW_RETURN_NOT_OK(AddStringBKDRHashFunction());
    ARROW_RETURN_NOT_OK(AddBKDRHashCombineFunction());
    return arrow::Status::OK();
}

} // namespace metaspore::serving