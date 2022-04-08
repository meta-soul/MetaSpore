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

package com.dmetasoul.metaspore.demo.movielens.retrieval.impl;

import com.dmetasoul.metaspore.demo.movielens.retrieval.RetrievalService;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.retrieval.matcher.Matcher;
import com.dmetasoul.metaspore.demo.movielens.retrieval.matcher.MatcherProvider;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

@Service
public class RetrievalServiceImpl implements RetrievalService {
    public static final int DEFAULT_MAX_RESERVATION = 500;
    public static final List<String> DEFAULT_MATCHER_NAMES = List.of("ItemCfMatcher");

    private final MatcherProvider matcherProvider;

    public RetrievalServiceImpl(MatcherProvider matcherProvider) {
        this.matcherProvider = matcherProvider;
    }

    @Override
    public List<ItemModel> match(RecommendContext recommendContext,
                                 UserModel userModel) throws IOException {
        List<String> matcherNames = recommendContext.getMatcherNames();
        if (matcherNames == null || matcherNames.size() == 0) {
            matcherNames = DEFAULT_MATCHER_NAMES;
        }
        Integer maxReservation = recommendContext.getRetrievalMaxReservation();
        if (maxReservation == null || maxReservation < 0) {
            maxReservation = DEFAULT_MAX_RESERVATION;
        }

        List<ItemModel> itemModels = new ArrayList<>();
        
        for (Matcher m : matcherProvider.getMatchers(matcherNames)) {
            itemModels.addAll(m.match(recommendContext, userModel));
        }

        // Merge duplicated items.
        itemModels = getMergedItemModels(itemModels);
        // Sort by final retrieval score.
        itemModels.sort((o1, o2) -> o2.getFinalRetrievalScore().compareTo(o1.getFinalRetrievalScore()));
        // Truncate
        if (itemModels.size() > maxReservation) {
            itemModels = itemModels.subList(0, maxReservation);
        }

        return itemModels;
    }

    private static List<ItemModel> getMergedItemModels(List<ItemModel> itemModels) {
        HashMap<String, ItemModel> itemModelMap = new HashMap<>();
        itemModels.forEach(x -> {
            if (itemModelMap.containsKey(x.getId())) {
                ItemModel itemModel = itemModelMap.get(x.getId());
                itemModel.getOriginalRetrievalScoreMap().putAll(x.getOriginalRetrievalScoreMap());
                if (x.getFinalRetrievalScore() > itemModel.getFinalRetrievalScore()) {
                    itemModel.setFinalRetrievalScore(x.getFinalRetrievalScore());
                }
            } else {
                itemModelMap.put(x.getId(), x);
            }
        });
        return new ArrayList<>(itemModelMap.values());
    }
}