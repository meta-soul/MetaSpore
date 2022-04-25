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

package com.dmetasoul.metaspore.demo.multimodal.algo.retrieval.impl;

import com.dmetasoul.metaspore.demo.multimodal.algo.retrieval.RetrievalService;
import com.dmetasoul.metaspore.demo.multimodal.algo.retrieval.matcher.Matcher;
import com.dmetasoul.metaspore.demo.multimodal.algo.retrieval.matcher.MatcherProvider;
import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class RetrievalServiceImpl implements RetrievalService {
    public static final List<String> DEFAULT_MATCHER_NAMES = List.of("ANNMatcher");

    private final MatcherProvider matcherProvider;

    public RetrievalServiceImpl(MatcherProvider matcherProvider) {
        this.matcherProvider = matcherProvider;
    }

    @Override
    public List<List<ItemModel>> match(SearchContext searchContext, QueryModel queryModel) throws IOException {
        List<List<ItemModel>> results = new ArrayList<>();

        Integer maxReservation = searchContext.getMatchMaxReservation();
        List<String> matcherNames = searchContext.getMatchMatcherNames();
        if (matcherNames == null || matcherNames.size() == 0) {
            matcherNames = DEFAULT_MATCHER_NAMES;
        }

        // run all of activate matchers and merge results
        Map<Integer, List<ItemModel>> matcherResults = new HashMap<>();
        for (Matcher m : matcherProvider.getMatchers(matcherNames)) {
            List<List<ItemModel>> allItemModels = m.match(searchContext, queryModel);
            for (int qid=0; qid<allItemModels.size(); qid++) {
                if (!matcherResults.containsKey(qid)) {
                    matcherResults.put(qid, new ArrayList<ItemModel>());
                }
                matcherResults.get(qid).addAll(allItemModels.get(qid));
            }
        }

        // dedup/sort/truncate results
        for (Integer qid : matcherResults.keySet()) {
            List<ItemModel> itemModels = matcherResults.get(qid);
            results.add(truncateItemModels(itemModels, maxReservation));
        }

        return results;
    }

    private static List<ItemModel> truncateItemModels(List<ItemModel> itemModels, Integer maxReservation) {
        // dedup
        Map<String, ItemModel> itemModelMap = new HashMap<>();
        for (ItemModel item : itemModels) {
            String itemId = item.getId();
            if (itemModelMap.containsKey(itemId)) {
                ItemModel other = itemModelMap.get(itemId);
                other.getOriginalRetrievalScoreMap().putAll(item.getOriginalRetrievalScoreMap());
                if (item.getFinalRetrievalScore() > other.getFinalRetrievalScore()) {
                    other.setFinalRetrievalScore(item.getFinalRetrievalScore());
                }
            } else {
                itemModelMap.put(itemId, item);
            }
        }

        // sort by desc
        itemModels = new ArrayList<>(itemModelMap.values());
        itemModels.sort((o1, o2) -> Double.compare(o2.getFinalRetrievalScore(), o1.getFinalRetrievalScore()));

        // truncate
        itemModels = itemModels.size() <= maxReservation ? itemModels : itemModels.subList(0, maxReservation);
        return itemModels;
    }
}
