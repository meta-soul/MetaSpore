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

package com.dmetasoul.metaspore.demo.multimodal.service;

import io.milvus.response.SearchResultsWrapper;

import java.util.List;
import java.util.Map;

public interface MilvusService {
    Map<String, String> getMilvusArgs();
    void setMilvusArgs(Map<String, String> args);
    Map<Integer, List<SearchResultsWrapper.IDScore>> findByEmbeddingVectors(List<List<Float>> vectors, int topK);
    Map<Integer, List<SearchResultsWrapper.IDScore>> findByEmbeddingVectors(List<List<Float>> vectors, int topK, long timeout);
}