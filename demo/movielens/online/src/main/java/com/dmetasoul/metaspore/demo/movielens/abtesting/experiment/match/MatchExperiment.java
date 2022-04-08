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

package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.match;

import com.dmetasoul.metaspore.pipeline.pojo.Context;
import com.google.common.collect.Lists;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.retrieval.RetrievalService;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
//import com.dmetasoul.metaspore.pipeline.impl.Context;
import lombok.SneakyThrows;
import org.springframework.stereotype.Component;

import java.util.*;

@ExperimentAnnotation(name = "match.base")
@Component
public class MatchExperiment implements BaseExperiment<RecommendResult, RecommendResult> {
    protected final RetrievalService retrievalService;

    protected List<String> matcherNames;

    protected Integer maxReservation;

    protected Integer itemCfAlgoLevel;

    protected Integer itemCfMaxReservation;

    public MatchExperiment(RetrievalService retrievalService) {
        this.retrievalService = retrievalService;
    }

    @Override
    public void initialize(Map<String, Object> args) {
        this.matcherNames = Lists.newArrayList(((LinkedHashMap<String, String>) args.getOrDefault("matcherNames", new LinkedHashMap<String, String>())).values());
        this.maxReservation = (Integer) args.get("maxReservation");
        this.itemCfAlgoLevel = (Integer) args.get("itemCfAlgoLevel");
        this.itemCfMaxReservation = (Integer) args.get("itemCfMaxReservation");

        System.out.println("match.base initialize... " + args);
    }

    @SneakyThrows
    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
        System.out.println("match.base experiment, userModel:" + recommendResult.getUserId());
        UserModel userModel = recommendResult.getUserModel();
        if (userModel == null) {
            System.out.println("match.base experiment, user model is null");
            return recommendResult;
        }

        RecommendContext recommendContext = recommendResult.getRecommendContext();
        recommendContext.setMatcherNames(matcherNames);
        recommendContext.setRetrievalMaxReservation(maxReservation);
        recommendContext.setItemCfAlgoLevel(itemCfAlgoLevel);
        recommendContext.setItemCfMaxReservation(itemCfMaxReservation);

        List<ItemModel> retrievalItemModels = retrievalService.match(recommendContext, userModel);
        recommendResult.setRecommendItemModels(retrievalItemModels);
        return recommendResult;
    }
}