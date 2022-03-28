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


import com.dmetasoul.metaspore.demo.movielens.domain.User;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class UserModel {
    private Double alpha = 1.0;
    private String userId;
    private String recentMovieIds;
    private String lastMovie;
    private String lastGenre;
    private Double userGreaterThanThreeRate;
    private Double userMovieAvgRating;
    private String[] recentMovieArr;
    private Map<String, Double> triggerWeightMap;
    private String splitor = "\u0001";

    public UserModel(User user) {
        this.userId = user.getUserId();
        this.recentMovieIds = user.getRecentMovieIds();
        this.lastMovie = user.getLastMovie();
        this.lastGenre = user.getLastGenre();
        this.userGreaterThanThreeRate = Optional.ofNullable(user.getUserGreaterThanThreeRate()).orElse(0.0);
        this.userMovieAvgRating = Optional.ofNullable(user.getUserMovieAvgRating()).orElse(0.0);

        initRecentTriggers(recentMovieIds);
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getRecentMovieIds() {
        return recentMovieIds;
    }

    public void setRecentMovieIds(String recentMovieIds) {
        this.recentMovieIds = recentMovieIds;
    }

    public String getLastMovie() {
        return lastMovie;
    }

    public void setLastMovie(String lastMovie) {
        this.lastMovie = lastMovie;
    }

    public String getLastGenre() {
        return lastGenre;
    }

    public void setLastGenre(String lastGenre) {
        this.lastGenre = lastGenre;
    }

    public Double getUserGreaterThanThreeRate() {
        return userGreaterThanThreeRate;
    }

    public void setUserGreaterThanThreeRate(Double userGreaterThanThreeRate) {
        this.userGreaterThanThreeRate = userGreaterThanThreeRate;
    }

    public Double getUserMovieAvgRating() {
        return userMovieAvgRating;
    }

    public void setUserMovieAvgRating(Double userMovieAvgRating) {
        this.userMovieAvgRating = userMovieAvgRating;
    }

    private void initRecentTriggers(String recentMovieIds) {
        Map<String, Double> itemToWight = new HashMap<>();
        recentMovieArr = recentMovieIds.split(splitor);
        for (int i = 0; i < recentMovieArr.length; i++) {
            itemToWight.put(recentMovieArr[i], 1 / (1 + Math.pow((recentMovieArr.length - i - 1), alpha)));
        }
        triggerWeightMap = itemToWight;
    }

    public String[] getRecentMovieArr() {
        return recentMovieArr;
    }

    public Map<String, Double> getTriggerWeightMap() {
        return triggerWeightMap;
    }
}