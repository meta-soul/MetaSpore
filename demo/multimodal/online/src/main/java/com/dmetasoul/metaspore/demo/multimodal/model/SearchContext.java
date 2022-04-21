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

package com.dmetasoul.metaspore.demo.multimodal.model;


import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.google.protobuf.ByteString;

import java.util.List;
import java.util.Map;

public class SearchContext {
    private String userId;

    private String qpQueryEmbeddingModelName;

    private String qpQueryProcessorModelName;

    private String matchEmbeddingModelName;

    private String matchEmbeddingVectorName;

    private List<String> matchMatcherNames;

    private Integer matchMaxReservation;

    private Map<String, String> matchMilvusArgs;

    private Integer rankMaxReservation;

    // QP results
    private Map<String, ByteString> qpResults;

    // retrieval results, item list
    private List<List<ItemModel>> matchItemModels;

    // ranking results
    private List<List<ItemModel>> rankItemModels;

    public SearchContext() {

    }

    public SearchContext(String userId) {
        this.userId = userId;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getQpQueryEmbeddingModelName() { return qpQueryEmbeddingModelName; }

    public void setQpQueryEmbeddingModelName(String modelName) { this.qpQueryEmbeddingModelName = modelName; }

    public String getQpQueryProcessorModelName() { return qpQueryProcessorModelName; }

    public void setQpQueryProcessorModelName(String processorName) { this.qpQueryProcessorModelName = processorName; }

    public String getMatchEmbeddingModelName() { return matchEmbeddingModelName; }

    public void setMatchEmbeddingModelName(String modelName) { this.matchEmbeddingModelName = modelName; }

    public String getMatchEmbeddingVectorName() { return this.matchEmbeddingVectorName; }

    public void setMatchEmbeddingVectorName(String matchEmbeddingVectorName) { this.matchEmbeddingVectorName = matchEmbeddingVectorName; }

    public List<String> getMatchMatcherNames() { return this.matchMatcherNames; }

    public void setMatchMatcherNames(List<String> matchMatcherNames) { this.matchMatcherNames = matchMatcherNames; }

    public Integer getMatchMaxReservation() { return  this.matchMaxReservation; }

    public void setMatchMaxReservation(Integer maxReservation) { this.matchMaxReservation = maxReservation; }

    public Map<String, String> getMatchMilvusArgs() { return this.matchMilvusArgs; }

    public void setMatchMilvusArgs(Map<String, String> milvusArgs) { this.matchMilvusArgs = milvusArgs; }

    public Integer getRankMaxReservation() { return rankMaxReservation; }

    public void setRankMaxReservation(Integer maxReservation) { this.rankMaxReservation = maxReservation; }

    public Map<String, ByteString> getQpResults() { return qpResults; }

    public void setQpResults(Map<String, ByteString> qpResults) { this.qpResults = qpResults; }

    public List<List<ItemModel>> getMatchItemModels() { return matchItemModels; }

    public void setMatchItemModels(List<List<ItemModel>> itemModels) { this.matchItemModels = itemModels; }

    public List<List<ItemModel>> getRankItemModels() { return rankItemModels; }

    public void setRankItemModels(List<List<ItemModel>> itemModels) { this.rankItemModels = itemModels; }

    @Override
    public String toString() {
        return "SearchContext{" +
                "userId='" + userId + '\'' +
                '}';
    }
}
