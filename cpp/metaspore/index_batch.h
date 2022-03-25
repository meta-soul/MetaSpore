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

#include <metaspore/pybind_utils.h>
#include <metaspore/string_utils.h>
#include <pybind11/numpy.h>

namespace metaspore {

class __attribute__((visibility("hidden"))) IndexBatch {
  public:
    IndexBatch(pybind11::list columns, const std::string &delimiters);

    const StringViewHashVector &GetCell(size_t i, size_t j, const std::string &column_name) const;

    pybind11::list ToList() const;

    size_t GetRows() const { return rows_; }
    size_t GetColumns() const { return split_columns_.size(); }

    std::string ToString() const;

  private:
    struct __attribute__((visibility("hidden"))) string_view_cell {
        StringViewHashVector items_;
        pybind11::object obj_;
    };

    using StringViewColumn = std::vector<string_view_cell>;

    static StringViewColumn SplitColumn(const pybind11::array &column, std::string_view delims);

    std::vector<StringViewColumn> split_columns_;
    size_t rows_;
};

} // namespace metaspore
