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

import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.data.IndexData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.util.JsonStringHashMap;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.springframework.util.Assert;

import java.util.*;

import static com.dmetasoul.metaspore.recommend.common.Utils.getObjectToMap;

@Data
@Slf4j
@ServiceAnnotation("ItemMatcher")
public class ItemMatcherTask extends AlgoTransformTask {
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;

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
        addFunction("toItemScore", (fields, result, config) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.size() > 0 && fields.get(0).isMatch(DataTypeEnum.STRING),
                    "toItemScore input[0] is recall userId string");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            List<String> userIds = fields.get(0).getValue();
            List<List<String>> recallItemData = null;
            List<List<Double>> recallWeights = null;
            List<Double> userProfileWeights = null;
            if (fields.size() > 2 && fields.get(1).isMatch(DataTypeEnum.LIST_STR) &&
                    fields.get(2).isMatch(DataTypeEnum.LIST_DOUBLE)) {
                recallItemData = fields.get(1).getValue();
                recallWeights = fields.get(2).getValue();
                if (fields.size() > 3 && fields.get(3).isMatch(DataTypeEnum.DOUBLE)) {
                    userProfileWeights = fields.get(3).getValue();
                }
            } else if (fields.size() > 1 && fields.get(1).isMatch(DataTypeEnum.LIST_STRUCT)){
                List<List<Object>> objData = fields.get(1).getValue();
                recallItemData = Lists.newArrayList();
                recallWeights = Lists.newArrayList();
                Field field = fields.get(1).getField();
                Assert.isTrue(field.getChildren().size() == 1, "list struct only has one struct children!");
                List<Field> children = field.getChildren().get(0).getChildren();
                Assert.isTrue(children != null && children.size() == 2, "itemscore must has 2 field!");
                Field itemField = children.get(0);
                Field scoreField = children.get(1);
                if (CollectionUtils.isNotEmpty(objData)) {
                    for (List<Object> itemData : objData) {
                        List<String> itemArray = Lists.newArrayList();
                        List<Double> scoreArray = Lists.newArrayList();
                        recallItemData.add(itemArray);
                        recallWeights.add(scoreArray);
                        if (CollectionUtils.isEmpty(itemData)) {
                            continue;
                        }
                        for (Object data : itemData) {
                            if (data == null) {
                                continue;
                            }
                            Map<String, Object> map;
                            if (data instanceof Map) {
                                map = (Map<String, Object>)data;
                            } else {
                                map = getObjectToMap(data);
                            }
                            itemArray.add((String) map.get(itemField.getName()));
                            scoreArray.add((Double) map.get(scoreField.getName()));
                        }
                    }
                }
            }
            Map<String, Map<String, Double>> UserItemScore = new HashMap<>();
            for (int i = 0; i< userIds.size(); ++i) {
                Map<String, Double> itemToItemScore = UserItemScore.computeIfAbsent(userIds.get(i), key->Maps.newHashMap());
                List<String> recallItem = recallItemData.get(i);
                if (CollectionUtils.isEmpty(recallItem)) continue;
                List<Double> recallWeight = recallWeights.get(i);
                Double userProfileWeight = Utils.get(userProfileWeights, i, 1.0);
                if (userProfileWeight == null) userProfileWeight = 1.0;
                for (int j = 0; j < recallItem.size(); ++j) {
                    String itemId = recallItem.get(j);
                    Double itemScore = Utils.get(recallWeight, j, 1.0) * userProfileWeight;
                    if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                        itemToItemScore.put(itemId, itemScore);
                    }
                }
            }
            int newIndex = 0;
            for (Map.Entry<String, Map<String, Double>> entry : UserItemScore.entrySet()) {
                result.get(0).addIndexData(FieldData.create(newIndex, entry.getKey()));
                result.get(1).addIndexData(FieldData.create(newIndex, entry.getValue()));
                newIndex += 1;
            }
            return true;
        });
        addFunction("toItemScore2", (fields, result, config) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.size() > 0 && fields.get(0).isMatch(DataTypeEnum.STRING),
                    "toItemScore2 input[0] is recall userId string");
            Assert.isTrue(fields.size() > 1 && fields.get(1).isMatch(DataTypeEnum.STRING),
                    "toItemScore2 input[1] is recall itemId string");
            Assert.isTrue(fields.size() > 2 && fields.get(2).isMatch(DataTypeEnum.DOUBLE),
                    "toItemScore2 input[2] is recall item score double");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            List<String> userIds = fields.get(0).getValue();
            List<String> recallItemData = fields.get(1).getValue();
            List<Double> recallWeights = fields.get(2).getValue();
            List<Double> userProfileWeights = null;
            if (fields.size() > 3 && fields.get(3).isMatch(DataTypeEnum.DOUBLE)) {
                userProfileWeights = fields.get(3).getValue();
            }
            Map<String, Map<String, Double>> UserItemScore = new HashMap<>();
            for (int i = 0; i< userIds.size(); ++i) {
                Map<String, Double> itemToItemScore = UserItemScore.computeIfAbsent(userIds.get(i), key->Maps.newHashMap());
                String itemId = recallItemData.get(i);
                Double itemScore = recallWeights.get(i) * Utils.get(userProfileWeights, i, 1.0);
                if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                    itemToItemScore.put(itemId, itemScore);
                }
            }
            int newIndex = 0;
            for (Map.Entry<String, Map<String, Double>> entry : UserItemScore.entrySet()) {
                result.get(0).addIndexData(FieldData.create(newIndex, entry.getKey()));
                result.get(1).addIndexData(FieldData.create(newIndex, entry.getValue()));
                newIndex += 1;
            }
            return true;
        });
        addFunction("recallCollectItem", (fields, result, config) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.size() > 0 && fields.get(0).isMatch(DataTypeEnum.STRING),
                    "recallCollectItem input[0] is recall userId string");
            Assert.isTrue(fields.get(1).isMatch(DataTypeEnum.MAP_STR_DOUBLE),
                    "recallCollectItem input[1] is userprofileWeight map<string, double>>");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            int limit = Utils.getField(options, "maxReservation", maxReservation);
            int finalAlgoLevel = Utils.getField(options, "algoLevel", algoLevel);
            List<IndexData> userIds = fields.get(0).getIndexValue();
            List<Map<String, Double>> itemScores = fields.get(1).getValue();
            for (int i = 0; i < userIds.size(); ++i) {
                Map<String, Double> itemScore = itemScores.get(i);
                if (MapUtils.isEmpty(itemScore)) continue;
                ArrayList<Map.Entry<String, Double>> entries = new ArrayList<>(itemScore.entrySet());
                entries.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
                Double maxScore = 0.0;
                if (itemScores.size() > 0) {
                    Map.Entry<String, Double> entry = entries.get(0);
                    maxScore = entry.getValue();
                }
                Double finalMaxScore = maxScore;
                long limit1 = maxReservation;
                for (Map.Entry<String, Double> x : entries) {
                    if (limit1-- == 0) break;
                    int index = userIds.get(i).getIndex();
                    result.get(0).addIndexData(FieldData.create(index, userIds.get(i).getVal()));
                    result.get(1).addIndexData(FieldData.create(index, x.getKey()));
                    result.get(2).addIndexData(FieldData.create(index, Utils.getFinalRetrievalScore(x.getValue(), finalMaxScore, finalAlgoLevel)));
                    result.get(3).addIndexData(FieldData.create(index, Map.of(algoName, x.getValue())));
                }
            }
            return true;
        });
    }
}
