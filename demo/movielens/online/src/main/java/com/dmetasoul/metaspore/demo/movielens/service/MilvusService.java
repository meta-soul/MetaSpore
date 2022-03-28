package com.dmetasoul.metaspore.demo.movielens.service;

import io.milvus.response.SearchResultsWrapper;

import java.util.List;
import java.util.Map;

public interface MilvusService {
    Map<Integer, List<SearchResultsWrapper.IDScore>> findByEmbeddingVectors(List<List<Float>> vectors, int topK);
    Map<Integer, List<SearchResultsWrapper.IDScore>> findByEmbeddingVectors(List<List<Float>> vectors, int topK, long timeout);
}
