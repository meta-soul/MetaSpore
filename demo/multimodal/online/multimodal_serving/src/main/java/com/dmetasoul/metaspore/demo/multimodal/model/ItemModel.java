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

import java.util.HashMap;
import java.util.Map;

public class ItemModel {
    protected String id;
    protected Map<String, Double> originalRetrievalScoreMap;
    protected Double finalRetrievalScore;
    protected Map<String, Double> originalRankingScoreMap;
    protected Double finalRankingScore;
    protected Double score;
    protected Map<String, Object> summary;

    public ItemModel() {
        originalRetrievalScoreMap = new HashMap<>();
        originalRankingScoreMap = new HashMap<>();
        summary = new HashMap<>();
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public Map<String, Double> getOriginalRetrievalScoreMap() { return originalRetrievalScoreMap; }

    public void setOriginalRetrievalScore(String algoName, Double score) {
        originalRetrievalScoreMap.put(algoName, score);
    }

    public Double getFinalRetrievalScore() { return this.finalRetrievalScore; }

    public void setFinalRetrievalScore(Double score) { finalRetrievalScore = score; }

    public Map<String, Double> getOriginalRankingScoreMap() { return originalRankingScoreMap; }

    public void setOriginalRankingScoreMap(String algoName, Double score) {
        this.originalRankingScoreMap.put(algoName, score);
    }

    public Double getFinalRankingScore() { return this.finalRankingScore; }

    public void setFinalRankingScore(Double score) { this.finalRankingScore = score; }

    public Double getScore() { return this.score; }

    public void setScore(Double score) { this.score = score; }

    public Map<String, Object> getSummary() { return this.summary; }

    public void setSummary(String name, Object value) {
        this.summary.put(name, value);
    }
}
