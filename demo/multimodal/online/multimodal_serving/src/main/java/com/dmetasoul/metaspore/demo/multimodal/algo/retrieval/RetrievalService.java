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

package com.dmetasoul.metaspore.demo.multimodal.algo.retrieval;

import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;

import java.io.IOException;
import java.util.List;

public interface RetrievalService {
    List<List<ItemModel>> match(SearchContext searchContext, QueryModel queryModel) throws IOException;
}
