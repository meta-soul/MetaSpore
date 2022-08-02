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
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Maps;
import io.milvus.response.SearchResultsWrapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@SuppressWarnings("rawtypes")
@Slf4j
@DataServiceAnnotation("MilvusSearch")
public class MilvusSearchTask extends AlgoTransformTask {
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;
    private int algoLevel;
    private int maxReservation;
    private String milvusTask;
    private String algoName;
    private String milvusItemTask;
    private String milvusCollectionName;
    private String embeddingTask;

    private String milvusIdCol;
    private String milvusItemIdCol;

    @Override
    public boolean initTask() {
        algoLevel = getOptionOrDefault("algoLevel", DEFAULT_ALGO_LEVEL);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        algoName = getOptionOrDefault("algo-name", "itemCF");
        milvusCollectionName = getOptionOrDefault("milvusCollectionName", "");
        milvusIdCol = getOptionOrDefault("milvusIdCol", "");
        milvusItemIdCol = getOptionOrDefault("milvusItemIdCol", "");
        for (String task : config.getDepend().getThen()) {
            FeatureConfig.AlgoTransform algoTransform = taskFlowConfig.getAlgoTransforms().get(task);
            if (algoTransform != null) {
                embeddingTask = task;
                continue;
            }
            FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(task);
            if (sourceTable != null && sourceTable.getKind().equals("milvus") && StringUtils.isNotEmpty(embeddingTask)) {
                milvusTask = task;
                continue;
            }
            if (sourceTable != null && StringUtils.isNotEmpty(milvusTask)) {
                milvusItemTask = task;
                if (StringUtils.isEmpty(milvusIdCol)) {
                    milvusIdCol = sourceTable.getColumnNames().get(0);
                    milvusItemIdCol = sourceTable.getColumnNames().get(1);
                }
            }
        }
        return !StringUtils.isEmpty(milvusItemTask) && !StringUtils.isEmpty(milvusCollectionName);
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        ServiceRequest req = super.makeRequest(depend, request, context);
        if (depend.equals(milvusTask)) {
            DataResult taskResult = getDataResultByName(embeddingTask, context);
            if (taskResult.getPredictResult() == null || taskResult.getPredictResult().getEmbedding() == null) {
                throw new RuntimeException("embedding gen fail at MilvusSearchTask!");
            }
            req.put("embedding", taskResult.getPredictResult().getEmbedding());
            req.put("collectionName", milvusCollectionName);
            req.setLimit(maxReservation);
        }
        if (depend.equals(milvusItemTask)) {
            DataResult taskResult = getDataResultByName(milvusTask, context);
            Map<Integer, List<SearchResultsWrapper.IDScore>> itemVectors = taskResult.getMilvusData();
            if (itemVectors.size() != 1) {
                // TODO error log
                throw new InvalidParameterException("Item vectors size must be 1. But got: " + itemVectors.size());
            }
            List<SearchResultsWrapper.IDScore> iDScores = itemVectors.values().iterator().next();
            // TODO if Milvus return list is sorted, we do not need to use sort it again.
            iDScores.sort((o1, o2) -> Double.compare(o2.getScore(), o1.getScore()));
            List<String> milvusIds = iDScores.stream().map(SearchResultsWrapper.IDScore::getLongID).map(String::valueOf).collect(Collectors.toList());
            req.put(milvusIdCol, milvusIds);
        }
        return req;
    }

        @Override
    public DataResult process(ServiceRequest request, DataContext context) {
            DataResult taskResult = getDataResultByName(milvusTask, context);
            Map<Integer, List<SearchResultsWrapper.IDScore>> itemVectors = taskResult.getMilvusData();
            List<SearchResultsWrapper.IDScore> iDScores = itemVectors.values().iterator().next();
            DataResult milvusItemResult = getDataResultByName(milvusItemTask, context);
            List<Map> milvusItemIds = milvusItemResult.getData();
            HashMap<String, String> milvusIdToItemIdMap = new HashMap<>();
            milvusItemIds.forEach(x -> milvusIdToItemIdMap.put(Utils.getField(x, milvusIdCol, ""), Utils.getField(x, milvusItemIdCol, "")));

            List<Map> itemModels = new ArrayList<>();
            Double maxScore = iDScores.size() > 0 ? iDScores.get(0).getScore() : 0.0;
            iDScores.forEach(x -> {
                Map<String, Object> item = Maps.newHashMap();
                if (Utils.setFieldFail(item, config.getColumnNames(), 0, milvusIdToItemIdMap.get(String.valueOf(x.getLongID())))) {
                    return;
                }
                Double score = (double) x.getScore();
                if (Utils.setFieldFail(item, config.getColumnNames(), 1, Utils.getFinalRetrievalScore(score, maxScore, algoLevel))) {
                    return;
                }
                Map<String, Object> value = Maps.newHashMap();
                value.put(algoName, score);
                if (Utils.setFieldFail(item, config.getColumnNames(), 2, value)) {
                    return;
                }
                itemModels.add(item);
            });
            DataResult result = new DataResult();
            result.setData(itemModels);
            return result;
    }
}
