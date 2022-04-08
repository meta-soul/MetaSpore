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

#include <common/hashmap/memory_mapped_array_hash_map.h>
#include <common/hashmap/memory_mapped_array_hash_map_loader.h>
#include <common/hashmap/multi_memory_mapped_map_search_helper.h>
#include <serving/inmem_sparse_lookup.h>
#include <serving/threadpool.h>
#include <serving/utils.h>

#include <fmt/format.h>
#include <range/v3/algorithm/any_of.hpp>

namespace metaspore::serving {

using KeyType = uint64_t;
using ValueType = float;
using MapType = metaspore::MemoryMappedArrayHashMap<KeyType, ValueType>;
using MapContainerType = std::vector<std::shared_ptr<MapType>>;
using namespace std::string_literals;

awaitable_status InMemorySparseLookupSource::load(const std::string &dir) {
    auto s = co_await boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [this, dir]() -> awaitable_status {
            size_t file_count = FileSystemHelpers::count_dat_files(dir);
            if (file_count == 0) {
                co_return absl::InvalidArgumentError(
                    fmt::format("SparseLookupModel to load dir path {} doesn't exist", dir));
            }
            hashmaps_.resize(file_count);
            for (auto const &dir_entry : std::filesystem::directory_iterator{dir}) {
                auto path = (std::filesystem::path)dir_entry;
                if (FileSystemHelpers::is_dat_file(path)) {
                    auto stem = path.stem().string();
                    ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
                        auto parsed_name, FileSystemHelpers::parse_dat_file_index(stem));
                    auto [file_num, file_index] = parsed_name;
                    if (file_num != file_count) {
                        co_return absl::DataLossError(
                            fmt::format("SparseLookupModel loading file {} with mismatched "
                                        "file count {} vs. {}",
                                        path.string(), parsed_name.second, file_count));
                    }
                    try {
                        MemoryMappedArrayHashMapLoader loader(path.string(),
                                                              /* disableMmap */ true);
                        hashmaps_[file_index] =
                            std::make_shared<MapType>(loader.get<KeyType, ValueType>());
                    } catch (const std::exception &e) {
                        co_return absl::InternalError(fmt::format(
                            "SparseLookupModel load file {} failed: {}", path.string(), e.what()));
                    }
                }
            }
            if (ranges::any_of(hashmaps_, [](auto &m) { return !m; })) {
                co_return absl::DataLossError(
                    fmt::format("SparseLookupModel load insufficient files from {}", dir));
            }
            vector_size_ = hashmaps_[0]->get_value_count_per_key();
            spdlog::info("SparseLookupModel loaded from dir {} with vector size {}", dir,
                         vector_size_);
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return s;
}

template <size_t N> struct AssignFromSearch {

    AssignFromSearch(ValueType *_values) : values(_values) {}

    void operator()(KeyType key, const ValueType *p, size_t count) {
        ValueType *target = values + count * N;
        if (p == nullptr) [[unlikely]] {
            static std::array<ValueType, N> v{ValueType()};
            VectorAssign<ValueType, N>::assign(v.data(), target);
        } else [[likely]] {
            VectorAssign<ValueType, N>::assign(p, target);
        }
    }

    ValueType *values;
};

template <> struct AssignFromSearch<0> {

    AssignFromSearch(ValueType *_values, size_t _vs) : values(_values), vector_size(_vs) {}

    void operator()(KeyType key, const ValueType *p, size_t count) {
        ValueType *target = values + count * vector_size;
        if (p == nullptr) [[unlikely]] {
            memset(target, 0, vector_size * sizeof(ValueType));
        } else [[likely]] {
            VectorAssign<ValueType, 0>::assign(p, target, vector_size);
        }
    }

    ValueType *values;
    size_t vector_size;
};

awaitable_result<std::shared_ptr<arrow::FloatTensor>>
InMemorySparseLookupSource::lookup(std::shared_ptr<arrow::UInt64Tensor> indices) {
    // get input indices shape and element num
    auto shape = indices->shape();
    auto orig_numelem = std::reduce(shape.begin(), shape.end(), 1L, std::multiplies<int64_t>());
    // get output values shape and element num
    shape.push_back(vector_size_);
    auto numelem = orig_numelem * vector_size_;

    // allocate output tensor
    ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto buffer, arrow::AllocateBuffer(numelem * sizeof(float)));
    ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
        auto output_tensor,
        arrow::FloatTensor::Make(std::shared_ptr<arrow::Buffer>(buffer.release()), shape));

    // search for keys and fill output tensor
    const KeyType *keys = (const KeyType *)indices->raw_data();
    ValueType *values = (ValueType *)output_tensor->raw_data();
    switch (vector_size_) {
    case 1:
        MultiMemoryMappedHashMapSearcher<KeyType, ValueType, MapContainerType, const KeyType *,
                                         AssignFromSearch<1>>::search(keys, keys + orig_numelem,
                                                                      hashmaps_,
                                                                      AssignFromSearch<1>(values));
        break;
    case 4:
        MultiMemoryMappedHashMapSearcher<KeyType, ValueType, MapContainerType, const KeyType *,
                                         AssignFromSearch<4>>::search(keys, keys + orig_numelem,
                                                                      hashmaps_,
                                                                      AssignFromSearch<4>(values));
        break;
        [[likely]] case 8 : MultiMemoryMappedHashMapSearcher<
                                KeyType, ValueType, MapContainerType, const KeyType *,
                                AssignFromSearch<8>>::search(keys, keys + orig_numelem, hashmaps_,
                                                             AssignFromSearch<8>(values));
        break;
        [[likely]] case 16 : MultiMemoryMappedHashMapSearcher<
                                 KeyType, ValueType, MapContainerType, const KeyType *,
                                 AssignFromSearch<16>>::search(keys, keys + orig_numelem, hashmaps_,
                                                               AssignFromSearch<16>(values));
        break;
    default:
        MultiMemoryMappedHashMapSearcher<KeyType, ValueType, MapContainerType, const KeyType *,
                                         AssignFromSearch<0>>::search(keys, keys + orig_numelem,
                                                                      hashmaps_,
                                                                      AssignFromSearch<0>(
                                                                          values, vector_size_));
    }
    co_return output_tensor;
}

awaitable_result<uint64_t> InMemorySparseLookupSource::get_vector_size() { co_return vector_size_; }

std::unique_ptr<SparseLookupModel::SparseLookupSource> InMemorySparseLookupSource::make() {
    return std::make_unique<InMemorySparseLookupSource>();
}

} // namespace metaspore::serving