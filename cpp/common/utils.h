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

#include <common/types.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <algorithm>
#include <filesystem>
#include <regex>

namespace metaspore {

template <typename T, size_t N> struct VectorAssign {
    inline static void assign(const T *from, T *to) {
        for (size_t i = 0; i < N; ++i) {
            to[i] = from[i];
        }
    }
};

template <typename T> struct VectorAssign<T, 1> {
    inline static void assign(const T *from, T *to) { to[0] = from[0]; }
};

template <typename T> struct VectorAssign<T, 0> {
    inline static void assign(const T *from, T *to, size_t n) { memcpy(to, from, n * sizeof(T)); }
};

class FileSystemHelpers {
  public:
    static bool is_dat_file(const std::filesystem::path &p) {
        using namespace std::string_literals;
        return p.extension().string() == ".dat"s;
    }

    static size_t count_dat_files(const std::string &dir) {
        size_t count = 0UL;
        std::filesystem::path dir_path(dir);
        if (!std::filesystem::is_directory(dir_path)) {
            return 0;
        }
        for (auto const &dir_entry : std::filesystem::directory_iterator{dir}) {
            auto path = (std::filesystem::path)dir_entry;
            if (is_dat_file(path)) {
                ++count;
            }
        }
        return count;
    }

    static result<std::pair<size_t, size_t>> parse_dat_file_index(const std::string &name) {
        static std::regex pattern(".*?([0-9]+)_([0-9]+)");
        std::smatch results;
        if (std::regex_match(name, results, pattern)) {
            if (results.size() == 3) {
                return std::make_pair(std::stoul(results[1]), std::stoul(results[2]));
            }
        }
        return absl::NotFoundError(
            fmt::format("SparseLookupModel load file with invalid name: {}", name));
    }
};

template <typename Func> struct Defer {
    Defer(Func &&func) : f(std::move(func)) {}

    ~Defer() { f(); }

    Func f;
};

} // namespace metaspore
