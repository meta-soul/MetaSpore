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
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.dmetasoul.metaspore.serving.PredictGrpc;
import com.dmetasoul.metaspore.serving.ServingClient;
import com.google.common.collect.Maps;
import io.milvus.response.SearchResultsWrapper;
import lombok.extern.slf4j.Slf4j;
import net.devh.boot.grpc.client.inject.GrpcClient;
import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@RecommendAnnotation("TwoTowersMatcher")
public class TwoTowersMatcher extends RecommendService {
    public static final String ALGO_NAME = "TwoTowersSimpleX";
    public static final String DEFAULT_MODEL_NAME = "two_towers_simplex";
    public static final String TARGET_KEY = "output";
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;

    //@GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;

    private int algoLevel;
    private int maxReservation;

    private String modelName;

    private String twoTowerUserTask;
    private String interactedItemsTask;
    private String milvusTask;
    private String milvusItemTask;

    private String milvusIdCol;
    private String milvusItemIdCol;

    @Override
    protected boolean initService() {
        algoLevel = getOptionOrDefault("algoLevel", DEFAULT_ALGO_LEVEL);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        twoTowerUserTask = getOptionOrDefault("twoTowerUser", "twoTowerUser");
        if (isInvalidDepend(twoTowerUserTask)) {
            return false;
        }
        interactedItemsTask = getOptionOrDefault("interactedItems", "interactedItems");
        if (isInvalidDepend(interactedItemsTask)) {
            return false;
        }
        milvusTask = getOptionOrDefault("milvus", "milvus");
        if (isInvalidDepend(milvusTask)) {
            return false;
        }
        milvusItemTask = getOptionOrDefault("milvusItem", "milvusItem");
        if (isInvalidDepend(milvusItemTask)) {
            return false;
        }
        milvusIdCol = getOptionOrDefault("idCol", getDependKey(milvusItemTask, 1));
        milvusItemIdCol = getOptionOrDefault("itemidCol", getDependKey(milvusItemTask, 2));
        modelName = getOptionOrDefault("modelName", DEFAULT_MODEL_NAME);
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {

        // 1. Prepare features.
        FeatureTable userTable = service.execute(twoTowerUserTask, context).getFeatureTable();
        FeatureTable interactedItemsTable = service.execute(interactedItemsTask, context).getFeatureTable();

        // 2. Call NSP service to get user vector.
        List<List<Float>> userVectors = getUserVectorsFromNps(modelName, List.of(userTable, interactedItemsTable));

        // 3. Call Milvus service to get top k item milvus ids.
        List<SearchResultsWrapper.IDScore> iDScores = getIDScoresFromMilvus(userVectors, maxReservation, context);

        // 4. Call MongoDB to get item ids from milvus ids.
        List<Map> milvusItemIds = getItemIdsFromCache(iDScores, context);

        // 5. Return ItemModel list as result.
        return getItemModels(iDScores, milvusItemIds, algoLevel);
    }

    private List<List<Float>> getUserVectorsFromNps(String modelName, List<FeatureTable> featureTables){
        Map<String, ArrowTensor> npsResultMap = null;
        try {
            npsResultMap = ServingClient.predictBlocking(client, modelName, featureTables, Collections.emptyMap());
        } catch (IOException e) {
            log.error("TwoTower request nps fail!");
            throw new RuntimeException(e);
        }
        return Utils.getVectorsFromNpsResult(npsResultMap, TARGET_KEY);
    }

    private List<SearchResultsWrapper.IDScore> getIDScoresFromMilvus(List<List<Float>> vectors, Integer maxReservation, DataContext context) {
        ServiceRequest req = new ServiceRequest(twoTowerUserTask, service.getName());
        req.put("embedding", vectors);
        req.setLimit(maxReservation);
        DataResult result = service.execute(twoTowerUserTask, req, context);
        Map<Integer, List<SearchResultsWrapper.IDScore>> itemVectors = result.getMilvusData();
        if (itemVectors.size() != 1) {
            // TODO error log
            throw new InvalidParameterException("Item vectors size must be 1. But got: " + itemVectors.size());
        }
        List<SearchResultsWrapper.IDScore> iDScores = itemVectors.values().iterator().next();
        // TODO if Milvus return list is sorted, we do not need to use sort it again.
        iDScores.sort((o1, o2) -> Double.compare(o2.getScore(), o1.getScore()));
        return iDScores;
    }

    private List<Map> getItemIdsFromCache(List<SearchResultsWrapper.IDScore> iDScores, DataContext context) {
        ServiceRequest req = new ServiceRequest(milvusTask, service.getName());
        List<String> milvusIds = iDScores.stream().map(SearchResultsWrapper.IDScore::getLongID).map(String::valueOf).collect(Collectors.toList());
        req.putIn(milvusIdCol, milvusIds);
        return service.execute(milvusTask, req, context).getData();
    }

    private DataResult getItemModels(List<SearchResultsWrapper.IDScore> iDScores, List<Map> milvusItemIds, Integer algoLevel) {
        // TODO if MongoDB return list is sequentially stable, we do not need to use HashMap here.
        HashMap<String, String> milvusIdToItemIdMap = new HashMap<>();
        milvusItemIds.forEach(x -> milvusIdToItemIdMap.put(getField(x, milvusIdCol, ""), getField(x, milvusItemIdCol, "")));

        List<Map> itemModels = new ArrayList<>();
        Double maxScore = iDScores.size() > 0 ? iDScores.get(0).getScore() : 0.0;
        iDScores.forEach(x -> {
            Map<String, Object> item = Maps.newHashMap();
            if (setFieldFail(item, 0, milvusIdToItemIdMap.get(String.valueOf(x.getLongID())))) {
                return;
            }
            Double score = (double) x.getScore();
            if (setFieldFail(item, 1, Utils.getFinalRetrievalScore(score, maxScore, algoLevel))) {
                return;
            }
            Map<String, Object> value = Maps.newHashMap();
            value.put(ALGO_NAME, score);
            if (setFieldFail(item, 2, value)) {
                return;
            }
            itemModels.add(item);
        });
        DataResult result = new DataResult();
        result.setData(itemModels);
        return result;
    }
}