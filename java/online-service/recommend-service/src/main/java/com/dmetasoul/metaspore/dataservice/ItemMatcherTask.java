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
package com.dmetasoul.metaspore.dataservice;

import com.dmetasoul.metaspore.common.Utils;
import com.dmetasoul.metaspore.configure.FieldInfo;
import com.dmetasoul.metaspore.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.common.CommonUtils;
import com.dmetasoul.metaspore.data.TableData;
import com.dmetasoul.metaspore.enums.DataTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.springframework.util.Assert;

import java.util.*;

@Data
@Slf4j
@ServiceAnnotation("ItemMatcher")
public class ItemMatcherTask extends AlgoTransformTask {
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 200;

    private int algoLevel;
    private int maxReservation;
    private String algoName;

    @Override
    public boolean initTask() {
        algoLevel = getOptionOrDefault("algoLevel", DEFAULT_ALGO_LEVEL);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        algoName = getOptionOrDefault("algo-name", "itemCF");
        return true;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void addFunctions() {
        addFunction("toItemScore", (fieldTableData, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(config.getInputFields() != null && config.getInputFields().size() > 2,
                    "recallCollectItem has three input");
            FieldInfo userId = config.getInputFields().get(0);
            FieldInfo userProfile = null;
            Map<String, Map<String, Double>> UserItemScore = new HashMap<>();
            for (int i = 0; i < fieldTableData.getData().size(); ++i) {
                Map<String, Double> itemToItemScore = UserItemScore.computeIfAbsent((String) fieldTableData.getValue(i, userId), key -> Maps.newHashMap());
                List<FieldInfo> input = config.getInputFields();
                List<String> recallItem = Lists.newArrayList();
                List<Double> recallWeight = Lists.newArrayList();
                if (input.size() > 1 && fieldTableData.getType(input.get(1)).equals(DataTypeEnum.LIST_STRUCT)) {
                    List<Object> itemData = (List<Object>) fieldTableData.getValue(i, input.get(1));
                    Field field = fieldTableData.getField(input.get(1));
                    Assert.isTrue(field.getChildren().size() == 1, "list struct only has one struct children!");
                    List<Field> children = field.getChildren().get(0).getChildren();
                    Assert.isTrue(children != null && children.size() == 2, "itemscore must has 2 field!");
                    Field itemField = children.get(0);
                    Field scoreField = children.get(1);
                    if (CollectionUtils.isEmpty(itemData)) {
                        continue;
                    }
                    for (Object data : itemData) {
                        if (data == null) {
                            continue;
                        }
                        Map<String, Object> map;
                        if (data instanceof Map) {
                            map = (Map<String, Object>) data;
                        } else {
                            map = CommonUtils.getObjectToMap(data);
                        }
                        recallItem.add((String) map.get(itemField.getName()));
                        recallWeight.add((Double) map.get(scoreField.getName()));
                    }
                    if (input.size() > 2) {
                        userProfile = input.get(2);
                    }
                } else if (input.size() > 2) {
                    recallItem.addAll((List<String>) fieldTableData.getValue(i, input.get(1)));
                    recallWeight.addAll((List<Double>) fieldTableData.getValue(i, input.get(2)));
                    if (input.size() > 3) {
                        userProfile = input.get(3);
                    }
                }
                double userProfileWeight = 1.0;
                if (userProfile != null) {
                    userProfileWeight = (double) fieldTableData.getValue(i, userProfile);
                }
                for (int j = 0; j < recallItem.size(); ++j) {
                    String itemId = recallItem.get(j);
                    Double itemScore = CommonUtils.get(recallWeight, j, 1.0) * userProfileWeight;
                    if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                        itemToItemScore.put(itemId, itemScore);
                    }
                }
            }
            List<String> names = config.getNames();
            List<Object> types = config.getTypes();
            TableData recallData = new TableData(names, types);
            List<Object> userIds = Lists.newArrayList();
            List<Object> scores = Lists.newArrayList();
            for (Map.Entry<String, Map<String, Double>> entry : UserItemScore.entrySet()) {
                userIds.add(entry.getKey());
                scores.add(entry.getValue());
            }
            Assert.isTrue(names.size() > 1, "output has 2 fields");
            recallData.addValueList(names.get(0), userIds);
            recallData.addValueList(names.get(1), scores);
            fieldTableData.reset(recallData);
            return true;
        });
        addFunction("toItemScore2", (fieldTableData, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(config.getInputFields() != null && config.getInputFields().size() > 2,
                    "recallCollectItem has three input");
            FieldInfo userId = config.getInputFields().get(0);
            FieldInfo recallItem = config.getInputFields().get(1);
            FieldInfo recallWeight = config.getInputFields().get(2);
            Map<String, Map<String, Double>> UserItemScore = new HashMap<>();
            for (int i = 0; i < fieldTableData.getData().size(); ++i) {
                Map<String, Double> itemToItemScore = UserItemScore.computeIfAbsent((String) fieldTableData.getValue(i, userId), key -> Maps.newHashMap());
                String itemId = (String) fieldTableData.getValue(i, recallItem);
                double userProfileWeight = 1.0;
                if (config.getInputFields().size() > 3
                        && DataTypeEnum.DOUBLE.equals(fieldTableData.getType(config.getInputFields().get(3)))) {
                    userProfileWeight = (double) fieldTableData.getValue(i, config.getInputFields().get(3));
                }
                Double itemScore = (double) fieldTableData.getValue(i, recallWeight) * userProfileWeight;
                if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                    itemToItemScore.put(itemId, itemScore);
                }
            }
            List<String> names = config.getNames();
            List<Object> types = config.getTypes();
            TableData recallData = new TableData(names, types);
            List<Object> itemIds = Lists.newArrayList();
            List<Object> scores = Lists.newArrayList();
            for (Map.Entry<String, Map<String, Double>> entry : UserItemScore.entrySet()) {
                itemIds.add(entry.getKey());
                scores.add(entry.getValue());
            }
            Assert.isTrue(names.size() > 1, "output has 2 fields");
            recallData.addValueList(names.get(0), itemIds);
            recallData.addValueList(names.get(1), scores);
            fieldTableData.reset(recallData);
            return true;
        });
        addFunction("recallCollectItem", (fieldTableData, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            int limit = CommonUtils.getField(options, "maxReservation", maxReservation);
            int finalAlgoLevel = CommonUtils.getField(options, "algoLevel", algoLevel);
            Assert.isTrue(config.getInputFields() != null && config.getInputFields().size() > 1,
                    "recallCollectItem has two input");
            FieldInfo userId = config.getInputFields().get(0);
            FieldInfo itemScores = config.getInputFields().get(1);
            List<Object> userIds = Lists.newArrayList();
            List<Object> itemIds = Lists.newArrayList();
            List<Object> scores = Lists.newArrayList();
            List<Object> originScores = Lists.newArrayList();
            List<String> names = config.getNames();
            List<Object> types = config.getTypes();
            Assert.isTrue(names.size() > 3, "output has 4 fields");
            TableData recallData = new TableData(names, types);
            for (int i = 0; i < fieldTableData.getData().size(); ++i) {
                Map<String, Double> itemScore = (Map<String, Double>) fieldTableData.getValue(i, itemScores);
                if (MapUtils.isEmpty(itemScore)) continue;
                ArrayList<Map.Entry<String, Double>> entries = new ArrayList<>(itemScore.entrySet());
                entries.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
                Double maxScore = 0.0;
                if (itemScore.size() > 0) {
                    Map.Entry<String, Double> entry = entries.get(0);
                    maxScore = entry.getValue();
                }
                Double finalMaxScore = maxScore;
                long limit1 = maxReservation;
                for (Map.Entry<String, Double> x : entries) {
                    if (limit1-- == 0) break;
                    userIds.add(fieldTableData.getValue(i, userId));
                    itemIds.add(x.getKey());
                    scores.add(Utils.getFinalRetrievalScore(x.getValue(), finalMaxScore, finalAlgoLevel));
                    originScores.add(Map.of(algoName, x.getValue()));
                }
            }
            recallData.addValueList(names.get(0), userIds);
            recallData.addValueList(names.get(1), itemIds);
            recallData.addValueList(names.get(2), scores);
            recallData.addValueList(names.get(3), originScores);
            fieldTableData.reset(recallData);
            return true;
        });
    }
}
