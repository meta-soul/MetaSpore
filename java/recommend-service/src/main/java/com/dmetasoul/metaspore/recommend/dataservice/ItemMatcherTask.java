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
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.FlatFunction;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.recommend.functions.ScatterFunction;
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

    @Override
    public void addFunctions() {
        addFunction("toItemScore", new FlatFunction() {
            /**
             *  生成召回的itemId->score数据
             *  后续计算只使用该函数生成结果 itemToItemScore
             */
            @SuppressWarnings("unchecked")
            @Override
            public List<Object> flat(List<Integer> indexs, List<FieldData> fields, Map<String, Object> options) {
                Assert.isTrue(CollectionUtils.isNotEmpty(fields) && indexs != null,
                        "input fields must not null");
                Assert.isTrue(fields.size() > 0 && fields.get(0).isMatch(DataTypeEnum.LIST_STR),
                        "toItemScore input[0] is recall itemId list<string>");
                Assert.isTrue(fields.size() > 1 && fields.get(1).isMatch(DataTypeEnum.LIST_DOUBLE),
                        "toItemScore input[1] is recall item weight list<double>");
                Map<String, Double> itemToItemScore = new HashMap<>();
                List<List<String>> recallItems = fields.get(0).getValue();
                List<List<Double>> recallWeights = fields.get(1).getValue();
                List<Double> userProfileWeights = null;
                if (fields.size() > 2 && fields.get(2).isMatch(DataTypeEnum.DOUBLE)) {
                    userProfileWeights = fields.get(2).getValue();
                }
                for (int i = 0; i< recallItems.size(); ++i) {
                    List<String> recallItem = recallItems.get(i);
                    List<Double> recallWeight = recallWeights.get(i);
                    double userProfileWeight = Utils.get(userProfileWeights, i, 1.0);
                    for (int j = 0; j < recallItem.size(); ++j) {
                        String itemId = recallItem.get(j);
                        Double itemScore = recallWeight.get(j) * userProfileWeight;
                        if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                            itemToItemScore.put(itemId, itemScore);
                        }
                    }
                }
                ArrayList<Map.Entry<String, Double>> entryList = new ArrayList<>(itemToItemScore.entrySet());
                entryList.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
                List<Object> res = Lists.newArrayList();
                res.addAll(entryList);
                return res;
            }
        });
        addFunction("toItemScore2", new Function() {
            /**
             *  生成召回的itemId->score数据
             *  后续计算只使用该函数生成结果 itemToItemScore
             */
            @SuppressWarnings("unchecked")
            @Override
            public List<Object> process(List<FieldData> fields, Map<String, Object> options) {
                Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                        "input fields must not null");
                Assert.isTrue(fields.size() > 0 && fields.get(0).isMatch(DataTypeEnum.STRING),
                        "toItemScore input[0] is recall itemId string");
                Assert.isTrue(fields.size() > 1 && fields.get(1).isMatch(DataTypeEnum.DOUBLE),
                        "toItemScore input[1] is recall item weight double");
                Map<String, Double> itemToItemScore = new HashMap<>();
                List<String> recallItem = fields.get(0).getValue();
                List<Double> recallWeight = fields.get(1).getValue();
                List<Double> userProfileWeights = null;
                if (fields.size() > 2 && fields.get(2).isMatch(DataTypeEnum.DOUBLE)) {
                    userProfileWeights = fields.get(2).getValue();
                }
                double userProfileWeight = Utils.get(userProfileWeights, 0, 1.0);
                for (int j = 0; j < recallItem.size(); ++j) {
                    String itemId = recallItem.get(j);
                    Double itemScore = recallWeight.get(j) * userProfileWeight;
                    if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                        itemToItemScore.put(itemId, itemScore);
                    }
                }
                ArrayList<Map.Entry<String, Double>> entryList = new ArrayList<>(itemToItemScore.entrySet());
                entryList.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
                List<Object> res = Lists.newArrayList();
                res.addAll(entryList);
                return res;
            }
        });
        addFunction("recallCollectItem", (ScatterFunction) (fields, names, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.get(0).isMatch(DataTypeEnum.LIST_OBJ),
                    "toItemScore input[0] is userprofileWeight list<entry<string, double>>");
            Assert.isTrue(CollectionUtils.isNotEmpty(names) && names.size() == 3,
                    "toItemScore names should = {itemId, score, originalScores}");
            int limit = Utils.getField(options, "maxReservation", maxReservation);
            int finalAlgoLevel = Utils.getField(options, "algoLevel", algoLevel);
            List<Object> itemScores = fields.get(0).getValue();
            Double maxScore = 0.0;
            if (itemScores.size() > 0) {
                Map.Entry<String, Double> entry = (Map.Entry<String, Double>) itemScores.get(0);
                maxScore = entry.getValue();
            }
            Map<String, List<Object>> res = Maps.newHashMap();
            Double finalMaxScore = maxScore;
            fields.stream().limit(maxReservation).forEach(x -> {
                Map.Entry<String, Double> entry = (Map.Entry<String, Double>) x;
                res.computeIfAbsent(names.get(0), k->Lists.newArrayList()).add(entry.getKey());
                res.computeIfAbsent(names.get(1), k->Lists.newArrayList()).add(Utils.getFinalRetrievalScore(entry.getValue(), finalMaxScore, finalAlgoLevel));
                res.computeIfAbsent(names.get(2), k->Lists.newArrayList()).add(Map.of(algoName, entry.getValue()));
            });
            return res;
        });
    }
}
