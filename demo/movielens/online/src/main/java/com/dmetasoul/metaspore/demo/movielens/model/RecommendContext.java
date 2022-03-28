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

import com.dmetasoul.metaspore.demo.movielens.model.request.PayloadParam;
import com.dmetasoul.metaspore.demo.movielens.ranking.ranker.RankingSortStrategy;

import java.util.List;

public class RecommendContext {
    private String userId;

    private List<String> matcherNames;

    private Integer retrievalMaxReservation;

    private Integer itemCfAlgoLevel;

    private Integer itemCfMaxReservation;

    private Integer swingAlgoLevel;

    private Integer swingMaxReservation;

    private String twoTowersSimpleXModelName;

    private Integer twoTowersSimpleXAlgoLevel;

    private Integer twoTowersSimpleXMaxReservation;

    private String rankerName;

    private RankingSortStrategy.Type rankingSortStrategyType;

    private Double rankingSortStrategyAlpha;

    private Double rankingSortStrategyBeta;

    private Integer rankingMaxReservation;

    private String lightGBMModelName;

    private String wideAndDeepModelName;

    private boolean useDebug = false;

    private boolean useDiversify = false;

    public RecommendContext(String userId) {
        this.userId = userId;
    }

    public RecommendContext(String userId, PayloadParam payloadParam) {
        this.userId = userId;
        this.useDebug = payloadParam.useDebug;
        this.useDiversify = payloadParam.useDiversify;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public List<String> getMatcherNames() {
        return matcherNames;
    }

    public void setMatcherNames(List<String> matcherNames) {
        this.matcherNames = matcherNames;
    }

    public Integer getRetrievalMaxReservation() {
        return retrievalMaxReservation;
    }

    public void setRetrievalMaxReservation(Integer retrievalMaxReservation) {
        this.retrievalMaxReservation = retrievalMaxReservation;
    }

    public Integer getItemCfAlgoLevel() {
        return itemCfAlgoLevel;
    }

    public void setItemCfAlgoLevel(Integer itemCfAlgoLevel) {
        this.itemCfAlgoLevel = itemCfAlgoLevel;
    }

    public Integer getItemCfMaxReservation() {
        return itemCfMaxReservation;
    }

    public void setItemCfMaxReservation(Integer itemCfMaxReservation) {
        this.itemCfMaxReservation = itemCfMaxReservation;
    }

    public Integer getSwingAlgoLevel() {
        return swingAlgoLevel;
    }

    public void setSwingAlgoLevel(Integer swingAlgoLevel) {
        this.swingAlgoLevel = swingAlgoLevel;
    }

    public Integer getSwingMaxReservation() {
        return swingMaxReservation;
    }

    public void setSwingMaxReservation(Integer swingMaxReservation) {
        this.swingMaxReservation = swingMaxReservation;
    }

    public String getTwoTowersSimpleXModelName() {
        return twoTowersSimpleXModelName;
    }

    public void setTwoTowersSimpleXModelName(String twoTowersSimpleXModelName) {
        this.twoTowersSimpleXModelName = twoTowersSimpleXModelName;
    }

    public Integer getTwoTowersSimpleXAlgoLevel() {
        return twoTowersSimpleXAlgoLevel;
    }

    public void setTwoTowersSimpleXAlgoLevel(Integer twoTowersSimpleXAlgoLevel) {
        this.twoTowersSimpleXAlgoLevel = twoTowersSimpleXAlgoLevel;
    }

    public Integer getTwoTowersSimpleXMaxReservation() {
        return twoTowersSimpleXMaxReservation;
    }

    public void setTwoTowersSimpleXMaxReservation(Integer twoTowersSimpleXMaxReservation) {
        this.twoTowersSimpleXMaxReservation = twoTowersSimpleXMaxReservation;
    }

    public String getRankerName() {
        return rankerName;
    }

    public void setRankerName(String rankerName) {
        this.rankerName = rankerName;
    }

    public RankingSortStrategy.Type getRankingSortStrategyType() {
        return rankingSortStrategyType;
    }

    public void setRankingSortStrategyType(RankingSortStrategy.Type rankingSortStrategyType) {
        this.rankingSortStrategyType = rankingSortStrategyType;
    }

    public Double getRankingSortStrategyAlpha() {
        return rankingSortStrategyAlpha;
    }

    public void setRankingSortStrategyAlpha(Double rankingSortStrategyAlpha) {
        this.rankingSortStrategyAlpha = rankingSortStrategyAlpha;
    }

    public Double getRankingSortStrategyBeta() {
        return rankingSortStrategyBeta;
    }

    public void setRankingSortStrategyBeta(Double rankingSortStrategyBeta) {
        this.rankingSortStrategyBeta = rankingSortStrategyBeta;
    }

    public Integer getRankingMaxReservation() {
        return rankingMaxReservation;
    }

    public void setRankingMaxReservation(Integer rankingMaxReservation) {
        this.rankingMaxReservation = rankingMaxReservation;
    }

    public String getLightGBMModelName() {
        return lightGBMModelName;
    }

    public void setLightGBMModelName(String lightGBMModelName) {
        this.lightGBMModelName = lightGBMModelName;
    }

    public String getWideAndDeepModelName() {
        return wideAndDeepModelName;
    }

    public void setWideAndDeepModelName(String wideAndDeepModelName) {
        this.wideAndDeepModelName = wideAndDeepModelName;
    }

    public boolean isUseDebug() {
        return useDebug;
    }

    public void setUseDebug(boolean useDebug) {
        this.useDebug = useDebug;
    }

    public boolean isUseDiversify() {
        return useDiversify;
    }

    public void setUseDiversify(boolean useDiversify) {
        this.useDiversify = useDiversify;
    }
}