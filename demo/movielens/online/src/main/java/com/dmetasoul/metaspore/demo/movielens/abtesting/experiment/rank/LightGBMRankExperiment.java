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

package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.rank;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.ranking.RankingService;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
//import com.dmetasoul.metaspore.pipeline.impl.Context;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import lombok.SneakyThrows;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "rank.lightGBM")
@Component
public class LightGBMRankExperiment extends RankExperiment {

    public LightGBMRankExperiment(RankingService rankingService) {
        super(rankingService);
    }

    @Override
    public void initialize(Map<String, Object> map) {
        System.out.println("rank.lightGBM initialize... " + map);
        super.initialize(map);
    }

    @SneakyThrows
    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
        System.out.println("rank.lightGBM experiment, userModel:" + recommendResult.getUserId());
        UserModel userModel = recommendResult.getUserModel();
        List<ItemModel> recommendItemModels = recommendResult.getRecommendItemModels();
        if (userModel == null) {
            System.out.println("rank.lightGBM experiment, user model is null");
            return recommendResult;
        }
        if (CollectionUtils.isEmpty(recommendItemModels)) {
            System.out.println("rank.lightGBM experiment, match result is null");
            return recommendResult;
        }

        // TODO set different ranking service according to the experiment settings
        RecommendContext recommendContext = recommendResult.getRecommendContext();
        recommendContext.setRankerName(this.rankerName);
        recommendContext.setRankingMaxReservation(this.maxReservation);
        recommendContext.setRankingSortStrategyType(this.sortStrategyType);
        recommendContext.setRankingSortStrategyAlpha(this.sortStrategyAlpha);
        recommendContext.setRankingSortStrategyBeta(this.sortStrategyBeta);
        recommendContext.setLightGBMModelName(this.modelName);

        List<ItemModel> rankingItemModels = rankingService.rank(recommendContext, userModel, recommendItemModels);
        recommendResult.setRecommendItemModels(rankingItemModels);
        return recommendResult;
    }
}