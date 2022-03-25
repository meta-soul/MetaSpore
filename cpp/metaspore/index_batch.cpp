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

#include <metaspore/index_batch.h>
#include <metaspore/stack_trace_utils.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace metaspore {

IndexBatch::IndexBatch(pybind11::list columns, const std::string &delimiters) {
    if (columns.empty()) {
        std::string serr;
        serr.append("empty columns list\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    split_columns_.reserve(columns.size());
    size_t rows = 0;
    for (size_t j = 0; j < columns.size(); j++) {
        pybind11::object item = columns[j];
        if (!pybind11::isinstance<pybind11::array>(item)) {
            std::string serr;
            serr.append("column " + std::to_string(j) + " is not numpy ndarray\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        pybind11::array arr = item.cast<pybind11::array>();
        if (arr.dtype().kind() != 'O') {
            std::string serr;
            serr.append("column " + std::to_string(j) + " is not numpy ndarray of object\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        StringViewColumn column = SplitColumn(arr, delimiters);
        if (j == 0)
            rows = column.size();
        else if (column.size() != rows) {
            std::string serr;
            serr.append("column " + std::to_string(j) +
                        " and column 0 are not of the same length; ");
            serr.append(std::to_string(column.size()) + " != " + std::to_string(rows));
            serr.append("\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        split_columns_.push_back(std::move(column));
    }
    if (rows == 0) {
        std::string serr;
        serr.append("number of rows is zero\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    rows_ = rows;
}

IndexBatch::StringViewColumn IndexBatch::SplitColumn(const pybind11::array &column,
                                                     std::string_view delims) {
    const size_t rows = column.size();
    StringViewColumn output;
    output.reserve(rows);
    for (size_t i = 0; i < rows; i++) {
        const void *item_ptr = column.data(i);
        // Consider avoiding complex casting here.
        PyObject *item = (PyObject *)(*(void **)item_ptr);
        pybind11::object cell = pybind11::reinterpret_borrow<pybind11::object>(item);
        auto [str, obj] = get_string_object_tuple(cell);
        auto items = SplitFilterStringViewHash(str, delims);
        output.push_back(string_view_cell{std::move(items), std::move(obj)});
    }
    return output;
}

const StringViewHashVector &IndexBatch::GetCell(size_t i, size_t j,
                                                const std::string &column_name) const {
    if (i >= rows_) {
        std::string serr;
        serr.append("row index i is out of range; ");
        serr.append(std::to_string(i) + " >= " + std::to_string(rows_));
        serr.append("\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    if (j >= split_columns_.size()) {
        std::string serr;
        serr.append("column index j (" + column_name + ") is out of range; ");
        serr.append(std::to_string(j) + " >= " + std::to_string(split_columns_.size()));
        serr.append("\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    auto &column = split_columns_.at(j);
    return column.at(i).items_;
}

pybind11::list IndexBatch::ToList() const {
    pybind11::list rows;
    for (size_t i = 0; i < rows_; i++) {
        pybind11::list row;
        for (size_t j = 0; j < split_columns_.size(); j++) {
            pybind11::list items;
            auto &cell = split_columns_.at(j).at(i).items_;
            for (size_t k = 0; k < cell.size(); k++) {
                auto item = pybind11::make_tuple(cell.at(k).view_, cell.at(k).hash_);
                items.append(item);
            }
            row.append(items);
        }
        rows.append(row);
    }
    return rows;
}

std::string IndexBatch::ToString() const {
    return "[IndexBatch: " + std::to_string(GetRows()) + " x " + std::to_string(GetColumns()) + "]";
}

} // namespace metaspore
