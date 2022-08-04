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
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.FlatFunction;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.util.*;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("ItemMatcher")
public class ItemMatcherTask extends AlgoTransformTask {
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;

    private int algoLevel;
    private int maxReservation;

    private String algoName;
    private String cfValuesCol;
    private String userProfileWeightCol;

    @Override
    public boolean initTask() {
        algoLevel = getOptionOrDefault("algoLevel", DEFAULT_ALGO_LEVEL);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        algoName = getOptionOrDefault("algo-name", "itemCF");
        userProfileWeightCol = getOptionOrDefault("weight", "userProfileWeight");
        cfValuesCol = getOptionOrDefault("cfValues", "cfValues");
        return true;
    }

    @Override
    public void addFunctions() {
        addFunction("toItemScore", new FlatFunction() {
            @Override
            public List<Object> flat(List<Integer> indexs, List<List<Object>> values, List<DataTypeEnum> types, Map<String, Object> options) {
                Assert.isTrue(CollectionUtils.isNotEmpty(values) && indexs != null, "input data is not null");
                Assert.isTrue(types.get(0).equals(DataTypeEnum.MAP_STR_OBJECT), "toItemScore type is map<string, object>");
                Map<String, Double> itemToItemScore = new HashMap<>();
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
                List<Object> res = Lists.newArrayList();
                int num = 0;
                for (int i = 0; i < input.size(); ++i) {
                    Object item = input.get(i);
                    Assert.isInstanceOf(Collection.class, item);
                    Collection<?> list = (Collection<?>) item;
                    for (Object o : list) {
                        num += 1;
                        indexs.add(i);
                        res.add(o);
                    }
                }
                return res;
            }
        });
        addFunction("recentWeight", new FlatFunction() {
            @Override
            public List<Object> flat(List<Integer> indexs, List<List<Object>> values, List<DataTypeEnum> types, Map<String, Object> options) {
                Assert.isTrue(CollectionUtils.isNotEmpty(values) && indexs != null, "input data is not null");
                List<Object> res = Lists.newArrayList();
                int num = 0;
                for (int i = 0; i < input.size(); ++i) {
                    Object item = input.get(i);
                    Assert.isInstanceOf(Collection.class, item);
                    Collection<?> list = (Collection<?>) item;
                    for (Object o : list) {
                        num += 1;
                        indexs.add(i);
                        res.add(1 / (1 + Math.pow((list.size() - i - 1), alpha)));
                    }
                }
                return res;
            }
        });
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult taskResult = getDataResultByName(config.getFeature().getThen().get(0), context);
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
            Utils.setFieldFail(item, config.getColumnNames(), 0, x.getKey());
            Utils.setFieldFail(item, config.getColumnNames(), 1, Utils.getFinalRetrievalScore(x.getValue(), maxScore, finalAlgoLevel));
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
