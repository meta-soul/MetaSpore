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


import com.google.protobuf.ByteString;

import java.util.Map;

public class SearchContext {
    private String userId;

    private String qpQueryEmbeddingModelName;

    private String qpQueryProcessorModelName;

    private String matchEmbeddingModelName;

    private Map<String, ByteString> qpResults;

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

    public Map<String, ByteString> getQpResults() { return qpResults; }

    public void setQpResults(Map<String, ByteString> qpResults) { this.qpResults = qpResults; }

    public String getMatchEmbeddingModelName() { return matchEmbeddingModelName; }

    public void setMatchEmbeddingModelName(String modelName) { this.matchEmbeddingModelName = modelName; }

    @Override
    public String toString() {
        return "SearchContext{" +
                "userId='" + userId + '\'' +
                '}';
    }
}
