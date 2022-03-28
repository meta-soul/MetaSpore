package com.dmetasoul.metaspore.demo.movielens.retrieval.matcher;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;

import java.io.IOException;
import java.util.List;

public interface Matcher {
    Double EPSILON = 0.001;

    List<ItemModel> match(RecommendContext recommendContext, UserModel userModel) throws IOException;
    static Double getFinalRetrievalScore(Double originalScore, Double maxScore, int algoLevel) {
        return originalScore / (maxScore + EPSILON) + algoLevel;
    }
}
