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
package com.dmetasoul.metaspore.recommend.dataservice;

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation
public abstract class ItemMatcherTask extends AlgoTransform {
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;

    private int algoLevel;
    private int maxReservation;

    private String user2itemTask;
    private String algoName;
    private String cfValuesCol;
    private String userProfileWeightCol;

    @Override
    public boolean initService() {
        config = taskFlowConfig.getAlgoTransforms().get(name);
        algoLevel = getOptionOrDefault("algoLevel", DEFAULT_ALGO_LEVEL);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        algoName = getOptionOrDefault("algo-name", "itemCF");
        user2itemTask = getOptionOrDefault("weight", "user2item");
        userProfileWeightCol = getOptionOrDefault("weight", "userProfileWeight");
        cfValuesCol = getOptionOrDefault("cfValues", "cfValues");
        Chain chain = new Chain();
        List<String> depends = List.of(user2itemTask);
        chain.setThen(depends);
        chains.add(chain);
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult taskResult = getDataResultByName(user2itemTask, context);
        List<Map> dataColumn = getDataByColumns(taskResult, List.of(cfValuesCol, userProfileWeightCol));
        HashMap<String, Double> itemToItemScore = new HashMap<>();
        for (Map<String, Object> item : dataColumn) {
            List itemCfValue = Utils.getField(item, cfValuesCol, Lists.newArrayList());
            double userProfileWeight = Utils.getField(item, userProfileWeightCol, 0.0);
            itemCfValue.forEach(x -> {
                Map<String, Object> map = (Map<String, Object>) x;
                String itemId = map.get("_1").toString();
                Double itemScore = (Double) map.get("_2") * userProfileWeight;
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
            if (Utils.setFieldFail(item, config.getColumnNames(), 0, x.getKey())) {
                return;
            }
            if (Utils.setFieldFail(item, config.getColumnNames(), 1, Utils.getFinalRetrievalScore(x.getValue(), maxScore, finalAlgoLevel))) {
                return;
            }
            Map<String, Object> value = Maps.newHashMap();
            value.put(algoName, x.getValue());
            if (Utils.setFieldFail(item, config.getColumnNames(), 2, value)) {
                return;
            }
            data.add(item);
        });
        DataResult result = new DataResult();
        result.setData(data);
        return result;
    }
}
