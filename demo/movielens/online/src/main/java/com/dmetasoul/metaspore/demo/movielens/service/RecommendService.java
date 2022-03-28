package com.dmetasoul.metaspore.demo.movielens.service;

import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;

import java.io.IOException;

public interface RecommendService {
    RecommendResult recommend(RecommendContext recommendContext) throws IOException;
}
