package com.dmetasoul.metaspore.demo.multimodal.model;

import java.util.HashMap;

public class ItemModel {
    protected String id;
    protected HashMap<String, Double> originalRetrievalScoreMap;
    protected Double finalRetrievalScore;

    public ItemModel() {
        originalRetrievalScoreMap = new HashMap<>();
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public HashMap<String, Double> getOriginalRetrievalScoreMap() { return originalRetrievalScoreMap; }

    public void setOriginalRetrievalScore(String algoName, Double score) {
        originalRetrievalScoreMap.put(algoName, score);
    }

    public Double getFinalRetrievalScore() { return this.finalRetrievalScore; }

    public void setFinalRetrievalScore(Double score) { finalRetrievalScore = score; }
}
