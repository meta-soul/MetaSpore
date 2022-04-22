package com.dmetasoul.metaspore.demo.multimodal.algo.retrieval;

import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;

import java.io.IOException;
import java.util.List;

public interface RetrievalService {
    List<List<ItemModel>> match(SearchContext searchContext, QueryModel queryModel) throws IOException;
}
