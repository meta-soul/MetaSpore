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