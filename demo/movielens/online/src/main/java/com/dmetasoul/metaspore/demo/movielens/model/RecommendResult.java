package com.dmetasoul.metaspore.demo.movielens.model;

import java.util.List;

public class RecommendResult {
    private String userId;

    private UserModel userModel;

    private RecommendContext recommendContext;

    private List<ItemModel> recommendItemModels;

    public RecommendResult() {
    }

    public RecommendResult(UserModel userModel, List<ItemModel> recommendItemModels) {
        this.userModel = userModel;
        this.recommendItemModels = recommendItemModels;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public UserModel getUserModel() {
        return userModel;
    }

    public void setUserModel(UserModel userModel) {
        this.userModel = userModel;
    }

    public RecommendContext getRecommendContext() {
        return recommendContext;
    }

    public void setRecommendContext(RecommendContext recommendContext) {
        this.recommendContext = recommendContext;
    }

    public List<ItemModel> getRecommendItemModels() {
        return recommendItemModels;
    }

    public void setRecommendItemModels(List<ItemModel> recommendItemModels) {
        this.recommendItemModels = recommendItemModels;
    }
}
