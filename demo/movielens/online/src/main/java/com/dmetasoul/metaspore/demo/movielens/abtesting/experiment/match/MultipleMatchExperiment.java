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

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.retrieval.RetrievalService;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import lombok.SneakyThrows;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "match.multiple")
@Component
public class MultipleMatchExperiment extends MatchExperiment {
    private Integer itemCfAlgoLevel;

    private Integer itemCfMaxReservation;

    private Integer swingAlgoLevel;

    private Integer swingMaxReservation;

    private String twoTowersSimpleXModelName;

    private Integer twoTowersSimpleXAlgoLevel;

    private Integer twoTowersSimpleXMaxReservation;

    public MultipleMatchExperiment(RetrievalService retrievalService) {
        super(retrievalService);
    }

    @Override
    public void initialize(Map<String, Object> args) {
        super.initialize(args);
        this.itemCfAlgoLevel = (Integer) args.get("itemCfAlgoLevel");
        this.itemCfMaxReservation = (Integer) args.get("itemCfMaxReservation");
        this.swingAlgoLevel = (Integer) args.get("swingAlgoLevel");
        this.swingMaxReservation = (Integer) args.get("swingMaxReservation");
        this.twoTowersSimpleXModelName = (String) args.get("twoTowersSimpleXModelName");
        this.twoTowersSimpleXAlgoLevel = (Integer) args.get("twoTowersSimpleXAlgoLevel");
        this.twoTowersSimpleXMaxReservation = (Integer) args.get("twoTowersSimpleXMaxReservation");

        System.out.println("match.multiple initialize... " + args);
    }

    @SneakyThrows
    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
        System.out.println("match.multiple experiment, userModel:" + recommendResult.getUserId());
        UserModel userModel = recommendResult.getUserModel();
        if (userModel == null) {
            System.out.println("match.multiple experiment, user model is null");
            return recommendResult;
        }

        RecommendContext recommendContext = recommendResult.getRecommendContext();
        recommendContext.setMatcherNames(matcherNames);
        recommendContext.setRetrievalMaxReservation(maxReservation);
        recommendContext.setItemCfAlgoLevel(itemCfAlgoLevel);
        recommendContext.setItemCfMaxReservation(itemCfMaxReservation);
        recommendContext.setSwingAlgoLevel(swingAlgoLevel);
        recommendContext.setSwingMaxReservation(swingMaxReservation);
        recommendContext.setTwoTowersSimpleXModelName(twoTowersSimpleXModelName);
        recommendContext.setTwoTowersSimpleXAlgoLevel(twoTowersSimpleXAlgoLevel);
        recommendContext.setTwoTowersSimpleXMaxReservation(twoTowersSimpleXMaxReservation);

        List<ItemModel> retrievalItemModels = retrievalService.match(recommendContext, userModel);
        recommendResult.setRecommendItemModels(retrievalItemModels);
        return recommendResult;
    }
}