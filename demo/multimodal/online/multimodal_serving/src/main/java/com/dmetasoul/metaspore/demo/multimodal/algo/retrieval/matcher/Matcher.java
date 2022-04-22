package com.dmetasoul.metaspore.demo.multimodal.algo.retrieval.matcher;

import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;

import java.util.List;
import java.io.IOException;

public interface Matcher {
    Double EPSILON = 0.001;

    List<List<ItemModel>> match(SearchContext searchContext, QueryModel queryModel) throws IOException;
    static Double getFinalRetrievalScore(Double originalScore, Double maxScore, int algoLevel) {
        return originalScore / (maxScore + EPSILON) + algoLevel;
    }
}
