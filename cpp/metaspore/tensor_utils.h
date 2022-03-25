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

#include <common/hashmap/data_types.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace metaspore {

size_t SliceElements(const std::vector<size_t> &shape);
size_t TotalElements(const std::vector<size_t> &shape);
std::string ShapeToString(const std::vector<size_t> &shape);
std::vector<size_t> ShapeFromString(const std::string &str);
void FillNaN(uint8_t *buffer, size_t size, DataType type);
void MakeInitializerReady(pybind11::object initializer);
void MakeUpdaterReady(pybind11::object udpater);

} // namespace metaspore
