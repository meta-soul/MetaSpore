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
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
@RecommendAnnotation("SwingMatcher")
public class SwingMatcher extends RecommendService {

    public static final String ALGO_NAME = "Swing";
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;

    private int algoLevel;
    private int maxReservation;

    private String userProfileTask;
    private String swingTask;

    private String swingIdsCol;
    private String userProfileWeightCol;

    @Override
    protected boolean initService() {
        algoLevel = getOptionOrDefault("algoLevel", DEFAULT_ALGO_LEVEL);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        userProfileTask = getOptionOrDefault("userProfile", "userProfile");
        if (isInvalidDepend(userProfileTask)) {
            return false;
        }
        swingTask = getOptionOrDefault("itemCf", "itemCf");
        if (isInvalidDepend(swingTask)) {
            return false;
        }
        userProfileWeightCol = getOptionOrDefault("weight", getDependKey(userProfileTask, 1));
        swingIdsCol = getOptionOrDefault("ids", getDependKey(swingTask, 1));
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult userProfile = service.execute(userProfileTask, context);
        ServiceRequest req = new ServiceRequest(swingTask, service.getName());
        String field = getDependKey(userProfileTask, 0);
        List<String> ids = userProfile.getData().stream()
                .map(item-> getField(item, field, ""))
                .filter(StringUtils::isNotEmpty).collect(Collectors.toList());
        req.putIn(field, ids);
        DataResult swings = service.execute(swingTask, req, context);
        Iterator<Map> recentMovieIt = userProfile.getData().iterator();
        Iterator<Map> swingIterator = swings.getData().iterator();
        HashMap<String, Double> itemToItemScore = new HashMap<>();
        while (swingIterator.hasNext() && recentMovieIt.hasNext()) {
            Map swing = swingIterator.next();
            List swingValue = getField(swing, swingIdsCol, Lists.newArrayList());
            Map recentItem = recentMovieIt.next();
            swingValue.forEach(x -> {
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