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
#include <common/test_utils.h>
#include <common/utils.h>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/testing/gtest_util.h>

using namespace metaspore;
using namespace metaspore::serving;
using namespace std::string_literals;

TEST(FeatureComputeFuncsTestSuite, TestStringBkdrHashFunc) {
    arrow::StringBuilder builder;
    arrow::UInt64Builder expectedBuilder;
    std::string column_name = "column_name";
    uint64_t seed = BKDRHashWithEqualPostfix(column_name.c_str(), column_name.length(), 0UL);
    for (int i = 0; i < 10; ++i) {
        std::string s = "test_123_"s + std::to_string(i);
        uint64_t hash = BKDRHash(s.c_str(), s.length(), 0);
        ASSERT_OK(builder.Append(s));
        ASSERT_OK(expectedBuilder.Append(BKDRHashOneField(seed, hash)));
        if (i % 2 == 1) {
            ASSERT_OK(builder.AppendNull());
            ASSERT_OK(expectedBuilder.AppendNull());
        }
    }
    ASSERT_OK_AND_ASSIGN(auto array, builder.Finish());
    ASSERT_OK_AND_ASSIGN(auto expected_array, expectedBuilder.Finish());
    StringBKDRHashFunctionOption option{column_name};
    ASSERT_OK_AND_ASSIGN(auto datum, arrow::compute::CallFunction("bkdr_hash", {array}, &option));
    ASSERT_TRUE(expected_array->Equals(datum.array_as<arrow::UInt64Array>()));
}

TEST(FeatureComputeFuncsTestSuite, TestStringListBkdrHashFunc) {
    auto string_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder string_list_builder(arrow::default_memory_pool(), string_builder,
                                           std::make_shared<arrow::ListType>(arrow::utf8()));
    auto uint64_builder = std::make_shared<arrow::UInt64Builder>();
    arrow::ListBuilder uint64_list_builder(arrow::default_memory_pool(), uint64_builder,
                                           std::make_shared<arrow::ListType>(arrow::uint64()));
    std::string column_name = "column_name";
    uint64_t seed = BKDRHashWithEqualPostfix(column_name.c_str(), column_name.length(), 0UL);
    for (int i = 0; i < 10; ++i) {
        ASSERT_OK(string_list_builder.Append());
        ASSERT_OK(uint64_list_builder.Append());
        for (int j = 0; j < i + 1; ++j) {
            std::string s = "test_123_"s + std::to_string(i) + "_"s + std::to_string(j);
            uint64_t hash = BKDRHash(s.c_str(), s.length(), 0);
            ASSERT_OK(string_builder->Append(s));
            ASSERT_OK(uint64_builder->Append(BKDRHashOneField(seed, hash)));
            if (j % 4 == 1) {
                ASSERT_OK(string_builder->AppendNull());
            }
        }
        if (i % 3 == 1) {
            ASSERT_OK(string_list_builder.AppendNull());
            ASSERT_OK(uint64_list_builder.AppendNull());
        }
    }
    ASSERT_OK_AND_ASSIGN(auto array, string_list_builder.Finish());
    ASSERT_OK_AND_ASSIGN(auto expected_array, uint64_list_builder.Finish());
    StringBKDRHashFunctionOption option{column_name};
    ASSERT_OK_AND_ASSIGN(auto datum, arrow::compute::CallFunction("bkdr_hash", {array}, &option));
    ASSERT_TRUE(expected_array->Equals(datum.array_as<arrow::ListArray>()));
}

TEST(FeatureComputeFuncsTestSuite, TestMultiStringListBkdrHashAndCombineFunc) {
    struct Builders {
        std::shared_ptr<arrow::StringBuilder> string_builder =
            std::make_shared<arrow::StringBuilder>();
        arrow::ListBuilder string_list_builder{arrow::default_memory_pool(), string_builder,
                                               std::make_shared<arrow::ListType>(arrow::utf8())};
        std::string column_name;
        uint64_t seed;

        Builders(const std::string &name)
            : column_name(name),
              seed(BKDRHashWithEqualPostfix(column_name.c_str(), column_name.length(), 0UL)) {}
    };
    std::shared_ptr<arrow::UInt64Builder> uint64_builder = std::make_shared<arrow::UInt64Builder>();
    arrow::ListBuilder uint64_list_builder{arrow::default_memory_pool(), uint64_builder,
                                           std::make_shared<arrow::ListType>(arrow::uint64())};
    std::vector<Builders> builders;
    for (size_t i = 0; i < 3; ++i) {
        builders.emplace_back(Builders("column_name_" + std::to_string(i)));
    }
    for (int i = 0; i < 5; ++i) {
        std::vector<uint64_t> v(6, 0);
        ASSERT_OK(uint64_list_builder.Append());
        for (size_t k = 0; k < 3; ++k) {
            auto &string_list_builder = builders[k].string_list_builder;
            auto &string_builder = builders[k].string_builder;
            auto seed = builders[k].seed;
            ASSERT_OK(string_list_builder.Append());
            for (int j = 0; j < k + 1; ++j) {
                std::string s = fmt::format("test_123_{}_{}_{}", k, i, j);
                uint64_t hash = BKDRHash(s.c_str(), s.length(), 0);
                hash = BKDRHashOneField(seed, hash);
                ASSERT_OK(string_builder->Append(s));
                if (k == 0) {
                    for (auto &hash_combined : v) {
                        hash_combined = hash;
                    }
                } else if (k == 1) {
                    for (size_t m = j * 3; m < j * 3 + 3; ++m) {
                        uint64_t &hash_combined = v[m];
                        hash_combined = BKDRHashConcatOneField(hash_combined, hash);
                    }
                } else {
                    {
                        uint64_t &hash_combined = v[j];
                        hash_combined = BKDRHashConcatOneField(hash_combined, hash);
                    }
                    {
                        uint64_t &hash_combined = v[j + 3];
                        hash_combined = BKDRHashConcatOneField(hash_combined, hash);
                    }
                }
                if (j % 2 == 1) {
                    ASSERT_OK(string_builder->AppendNull());
                }
            }
            if (i % 3 == 1) {
                ASSERT_OK(builders[k].string_list_builder.AppendNull());
            }
        }
        for (auto h : v) {
            ASSERT_OK(uint64_builder->Append(h));
        }
        if (i % 3 == 1) {
            ASSERT_OK(uint64_list_builder.AppendNull());
        }
    }
    std::vector<arrow::Datum> hash_outputs;
    for (size_t k = 0; k < 3; ++k) {
        ASSERT_OK_AND_ASSIGN(auto array, builders[k].string_list_builder.Finish());
        StringBKDRHashFunctionOption option{builders[k].column_name};
        ASSERT_OK_AND_ASSIGN(auto datum,
                             arrow::compute::CallFunction("bkdr_hash", {array}, &option));
        hash_outputs.push_back(std::move(datum));
    }
    ASSERT_OK_AND_ASSIGN(auto expected_array, uint64_list_builder.Finish());
    BKDRHashCombineFunctionOption option;
    ASSERT_OK_AND_ASSIGN(auto datum,
                         arrow::compute::CallFunction("bkdr_hash_combine", hash_outputs, &option));
    ASSERT_TRUE(expected_array->Equals(datum.array_as<arrow::ListArray>()));
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }