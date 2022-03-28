package com.dmetasoul.metaspore.demo.movielens.retrieval;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;

import java.io.IOException;
import java.util.List;

public interface RetrievalService {
    List<ItemModel> match(RecommendContext recommendContext, UserModel userModel) throws IOException;
}
