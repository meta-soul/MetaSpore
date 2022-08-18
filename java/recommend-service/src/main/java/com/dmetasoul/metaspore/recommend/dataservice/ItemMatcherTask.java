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
import org.apache.arrow.vector.util.JsonStringHashMap;
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
        addFunction("toItemScore", (fields, result, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.size() > 0 && fields.get(0).isMatch(DataTypeEnum.LIST_STR),
                    "toItemScore input[0] is recall itemId list<string>");
            Assert.isTrue(fields.size() > 1 && fields.get(1).isMatch(DataTypeEnum.LIST_DOUBLE),
                    "toItemScore input[1] is recall item weight list<double>");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            Map<String, Double> itemToItemScore = new HashMap<>();
            Map<String, Integer> itemToIndex = new HashMap<>();
            List<IndexData> recallItemData = fields.get(0).getIndexValue();
            List<List<Double>> recallWeights = fields.get(1).getValue();
            List<Double> userProfileWeights = null;
            if (fields.size() > 2 && fields.get(2).isMatch(DataTypeEnum.DOUBLE)) {
                userProfileWeights = fields.get(2).getValue();
            }
            for (int i = 0; i< recallItemData.size(); ++i) {
                List<String> recallItem = recallItemData.get(i).getVal();
                List<Double> recallWeight = recallWeights.get(i);
                double userProfileWeight = Utils.get(userProfileWeights, i, 1.0);
                for (int j = 0; j < recallItem.size(); ++j) {
                    String itemId = recallItem.get(j);
                    Double itemScore = recallWeight.get(j) * userProfileWeight;
                    if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                        itemToItemScore.put(itemId, itemScore);
                        itemToIndex.put(itemId, recallItemData.get(i).getIndex());
                    }
                }
            }
            ArrayList<Map.Entry<String, Double>> entryList = new ArrayList<>(itemToItemScore.entrySet());
            entryList.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
            for (Map.Entry<String, Double> entry : entryList) {
                int index = itemToIndex.get(entry.getKey());
                result.get(0).addIndexData(FieldData.create(index, entry));
            }
            return true;
        });
        addFunction("toItemScore2", (fields, result, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.size() > 0 && fields.get(0).isMatch(DataTypeEnum.STRING),
                    "toItemScore input[0] is recall itemId string");
            Assert.isTrue(fields.size() > 1 && fields.get(1).isMatch(DataTypeEnum.DOUBLE),
                    "toItemScore input[1] is recall item weight double");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            Map<String, Double> itemToItemScore = new HashMap<>();
            Map<String, Integer> itemToIndex = new HashMap<>();
            List<IndexData> recallItemData = fields.get(0).getIndexValue();
            List<Double> recallWeight = fields.get(1).getValue();
            List<Double> userProfileWeights = null;
            if (fields.size() > 2 && fields.get(2).isMatch(DataTypeEnum.DOUBLE)) {
                userProfileWeights = fields.get(2).getValue();
            }
            double userProfileWeight = Utils.get(userProfileWeights, 0, 1.0);
            for (int j = 0; j < recallItemData.size(); ++j) {
                String itemId = (String) recallItemData.get(j).getVal();
                Double itemScore = recallWeight.get(j) * userProfileWeight;
                if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                    itemToItemScore.put(itemId, itemScore);
                    itemToIndex.put(itemId, recallItemData.get(j).getIndex());
                }
            }
            ArrayList<Map.Entry<String, Double>> entryList = new ArrayList<>(itemToItemScore.entrySet());
            entryList.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
            for (Map.Entry<String, Double> entry : entryList) {
                int index = itemToIndex.get(entry.getKey());
                result.get(0).addIndexData(FieldData.create(index, entry));
            }
            return true;
        });
        addFunction("recallCollectItem", (fields, result, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.get(0).isMatch(DataTypeEnum.LIST_PAIR_STR_DOUBLE),
                    "toItemScore input[0] is userprofileWeight list<entry<string, double>>");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            int limit = Utils.getField(options, "maxReservation", maxReservation);
            int finalAlgoLevel = Utils.getField(options, "algoLevel", algoLevel);
            List<IndexData> itemScores = fields.get(0).getValue();
            Double maxScore = 0.0;
            if (itemScores.size() > 0) {
                Map<String, Object> entry = itemScores.get(0).getVal();
                maxScore = (Double) entry.get("value");
            }
            Map<String, List<Object>> res = Maps.newHashMap();
            Double finalMaxScore = maxScore;
            itemScores.stream().limit(maxReservation).forEach(x -> {
                Map<String, Object> entry = x.getVal();
                result.get(0).addIndexData(FieldData.create(x.getIndex(), entry.get("key")));
                result.get(1).addIndexData(FieldData.create(x.getIndex(), Utils.getFinalRetrievalScore((Double) entry.get("value"), finalMaxScore, finalAlgoLevel)));
                result.get(2).addIndexData(FieldData.create(x.getIndex(), Map.of(algoName, (Double) entry.get("value"))));
            });
            return true;
        });
    }
}
