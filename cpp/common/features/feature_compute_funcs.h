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

#include <arrow/compute/api.h>

namespace metaspore {

template <typename FunctionOption>
class MetaSporeBKDRHashFuncOptType : public arrow::compute::FunctionOptionsType {
  public:
    const char *type_name() const override { return FunctionOption::type_name().c_str(); }
    std::string Stringify(const arrow::compute::FunctionOptions &option) const override {
        return static_cast<const FunctionOption &>(option).to_string();
    }
    bool Compare(const arrow::compute::FunctionOptions &l,
                 const arrow::compute::FunctionOptions &r) const override {
        return static_cast<const FunctionOption &>(l) == static_cast<const FunctionOption &>(r);
    }
    std::unique_ptr<arrow::compute::FunctionOptions>
    Copy(const arrow::compute::FunctionOptions &other) const override {
        return std::make_unique<FunctionOption>(static_cast<const FunctionOption &>(other));
    }

    static MetaSporeBKDRHashFuncOptType *get() {
        static MetaSporeBKDRHashFuncOptType type;
        return &type;
    }
};

class StringBKDRHashFunctionOption : public arrow::compute::FunctionOptions {
  public:
    StringBKDRHashFunctionOption(const StringBKDRHashFunctionOption &) = default;
    StringBKDRHashFunctionOption();
    StringBKDRHashFunctionOption(const std::string &_name);
    bool operator==(const StringBKDRHashFunctionOption &other) const = default;
    std::string to_string() const { return "StringBKDRHashFunctionOption::name=" + name; }
    static const std::string &type_name() {
        static const std::string t = "StringBKDRHashFunctionOptionType";
        return t;
    }
    static std::shared_ptr<StringBKDRHashFunctionOption> Make(const std::string &name) {
        return std::make_shared<StringBKDRHashFunctionOption>(name);
    }

  public:
    std::string name;
};

class BKDRHashCombineFunctionOption : public arrow::compute::FunctionOptions {
  public:
    BKDRHashCombineFunctionOption(const BKDRHashCombineFunctionOption &) = default;
    BKDRHashCombineFunctionOption();
    bool operator==(const BKDRHashCombineFunctionOption &other) const = default;
    std::string to_string() const { return "BKDRHashCombineFunctionOption"; }
    static const std::string &type_name() {
        static const std::string t = "BKDRHashCombineFunctionOptionType";
        return t;
    }
    static std::shared_ptr<BKDRHashCombineFunctionOption> Make() {
        return std::make_shared<BKDRHashCombineFunctionOption>();
    }
};

arrow::Status RegisterAllFunctions();

} // namespace metaspore
