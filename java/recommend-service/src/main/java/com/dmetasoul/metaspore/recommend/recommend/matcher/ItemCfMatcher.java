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

package com.dmetasoul.metaspore.recommend.recommend.matcher;

import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.recommend.RecommendService;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@RecommendAnnotation("ItemCfMatcher")
public class ItemCfMatcher extends RecommendService {
    public static final String ALGO_NAME = "ItemCF";
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;

    private int algoLevel;
    private int maxReservation;

    private String userProfileTask;
    private String itemCfTask;

    private String itemCfIdsCol;
    private String userProfileWeightCol;



    @Override
    protected boolean initService() {
        algoLevel = getOptionOrDefault("algoLevel", DEFAULT_ALGO_LEVEL);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        userProfileTask = getOptionOrDefault("userProfile", "userProfile");
        if (isInvalidDepend(userProfileTask)) {
            return false;
        }
        itemCfTask = getOptionOrDefault("itemCf", "itemCf");
        if (isInvalidDepend(itemCfTask)) {
            return false;
        }
        userProfileWeightCol = getOptionOrDefault("weight", getDependKey(userProfileTask, 1));
        itemCfIdsCol = getOptionOrDefault("ids", getDependKey(itemCfTask, 1));
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult userProfile = service.execute(userProfileTask, context);
        ServiceRequest req = new ServiceRequest(itemCfTask, service.getName());
        String field = getDependKey(userProfileTask, 0);
        List<String> ids = userProfile.getData().stream()
                .map(item-> getField(item, field, ""))
                .filter(StringUtils::isNotEmpty).collect(Collectors.toList());
        req.putIn(field, ids);
        DataResult itemCf = service.execute(itemCfTask, req, context);
        Iterator<Map> recentMovieIt = userProfile.getData().iterator();
        Iterator<Map> itemCfIterator = itemCf.getData().iterator();
        HashMap<String, Double> itemToItemScore = new HashMap<>();
        while (itemCfIterator.hasNext() && recentMovieIt.hasNext()) {
            Map itemcf = itemCfIterator.next();
            List itemCfValue = getField(itemcf, itemCfIdsCol, Lists.newArrayList());
            Map recentItem = recentMovieIt.next();
            itemCfValue.forEach(x -> {
                Map<String, Object> map = (Map<String, Object>) x;
                String itemId = map.get("_1").toString();
                Double itemScore = (Double) map.get("_2") * getField(recentItem, userProfileWeightCol, 0.0);
                if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                    itemToItemScore.put(itemId, itemScore);
                }
            });
        }
        ArrayList<Map.Entry<String, Double>> entryList = new ArrayList<>(itemToItemScore.entrySet());
        entryList.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        Double maxScore = entryList.size() > 0 ? entryList.get(0).getValue() : 0.0;
        Integer finalAlgoLevel = algoLevel;
        List<Map> data = Lists.newArrayList();
        entryList.stream().limit(maxReservation).forEach(x -> {
            Map<String, Object> item = Maps.newHashMap();
            if (setFieldFail(item, 0, x.getKey())) {
                return;
            }
            if (setFieldFail(item, 1, Utils.getFinalRetrievalScore(x.getValue(), maxScore, finalAlgoLevel))) {
                return;
            }
            Map<String, Object> value = Maps.newHashMap();
            value.put(ALGO_NAME, x.getValue());
            if (setFieldFail(item, 2, value)) {
                return;
            }
            data.add(item);
        });
        DataResult result = new DataResult();
        result.setData(data);
        return result;
    }
}