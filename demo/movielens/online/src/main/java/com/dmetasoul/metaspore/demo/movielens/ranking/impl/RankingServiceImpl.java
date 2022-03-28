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

package com.dmetasoul.metaspore.demo.movielens.ranking.impl;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.ranking.RankingService;
import com.dmetasoul.metaspore.demo.movielens.ranking.ranker.RankerProvider;
import com.dmetasoul.metaspore.demo.movielens.ranking.ranker.RankingSortStrategy;
import com.dmetasoul.metaspore.demo.movielens.domain.ItemFeature;
import com.dmetasoul.metaspore.demo.movielens.ranking.ranker.Ranker;
import com.dmetasoul.metaspore.demo.movielens.repository.ItemFeatureRepository;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class RankingServiceImpl implements RankingService {
    public static final int DEFAULT_MAX_RESERVATION = 50;
    public static final String DEFAULT_RANKER_NAME = "WideAndDeepRanker";

    // private final List<Ranker> rankers;
    private final RankerProvider rankerProvider;
    private final ItemFeatureRepository itemFeatureRepository;

    public RankingServiceImpl(RankerProvider rankerProvider, ItemFeatureRepository itemFeatureRepository) {
        this.rankerProvider = rankerProvider;
        this.itemFeatureRepository = itemFeatureRepository;
    }

    @Override
    public List<ItemModel> rank(RecommendContext recommendContext, UserModel userModel, List<ItemModel> itemModels) throws IOException {
        Ranker ranker = rankerProvider.getRanker(recommendContext.getRankerName());
        if (ranker == null) {
            ranker = rankerProvider.getRanker(DEFAULT_RANKER_NAME);
        }

        Integer maxReservation = recommendContext.getRankingMaxReservation();
        if (maxReservation == null || maxReservation <= 0) {
            maxReservation = DEFAULT_MAX_RESERVATION;
        }

        fillItemFeatures(itemModels);
        itemModels = ranker.rank(recommendContext, userModel, itemModels);

        // Sort with strategy
        itemModels.forEach(x -> {
            Double rankingScore = 0.0;
            if (x.getOriginalRankingScoreMap() != null && x.getOriginalRankingScoreMap().size() == 1) {
                rankingScore = x.getOriginalRankingScoreMap().values().iterator().next();
            }
            Double finalRankingScore = RankingSortStrategy.getScoreByStrategy(
                    recommendContext.getRankingSortStrategyType(),
                    x.getFinalRetrievalScore(),
                    recommendContext.getRankingSortStrategyAlpha(),
                    rankingScore,
                    recommendContext.getRankingSortStrategyBeta()
            );
            x.setFinalRankingScore(finalRankingScore);
        });
        itemModels.sort((o1, o2) -> o2.getFinalRankingScore().compareTo(o1.getFinalRankingScore()));

        // Truncate
        if (itemModels.size() > maxReservation) {
            itemModels = itemModels.subList(0, maxReservation);
        }

        return itemModels;
    }

    private void fillItemFeatures(List<ItemModel> itemModels) {
        List<String> itemIds = itemModels.stream().map(ItemModel::getId).collect(Collectors.toList());
        Collection<ItemFeature> itemFeatures = itemFeatureRepository.findByQueryidIn(itemIds);
        // TODO if MongoDB return list is sequentially stable, we do not need to use HashMap here.
        HashMap<String, ItemFeature> itemFeatureMap = new HashMap<>();
        itemFeatures.forEach(x -> itemFeatureMap.put(x.getQueryid(), x));
        itemModels.forEach(x -> x.fillFeatures(itemFeatureMap.get(x.getId())));
    }
}