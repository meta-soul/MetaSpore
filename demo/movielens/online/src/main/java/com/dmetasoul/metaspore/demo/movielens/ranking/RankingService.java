package com.dmetasoul.metaspore.demo.movielens.ranking;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;

import java.io.IOException;
import java.util.List;

public interface RankingService {
    List<ItemModel> rank(RecommendContext recommendContext, UserModel userModel, List<ItemModel> itemModels) throws IOException;
}
