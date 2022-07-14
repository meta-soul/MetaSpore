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

package com.dmetasoul.metaspore.recommend.recommend.ranker;

import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.recommend.RecommendService;
import com.dmetasoul.metaspore.serving.*;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@RecommendAnnotation("LightGBMRanker")
public class LightGBMRanker extends RecommendService {
    public static final String ALGO_NAME = "LightGBM";
    public static final String DEFAULT_MODEL_NAME = "lightgbm_test_model";
    public static final String TARGET_KEY = "probabilities";
    public static final int TARGET_INDEX = 1;

    //@GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;
    private String modelName;

    private String itemIdCol;

    private String lightGBMFeatureTask;


    @Override
    protected boolean initService() {
        modelName = getOptionOrDefault("modelName", DEFAULT_MODEL_NAME);
        itemIdCol = getOptionOrDefault("itemId", "itemId");
        lightGBMFeatureTask = getOptionOrDefault("lightGBM", "lightGBM");
        if (isInvalidDepend(lightGBMFeatureTask)) {
            return false;
        }
        itemIdCol = getOptionOrDefault("itemId", "itemId");
        return true;
    }

    // TODO maybe we should add a parent class for all ranker implements like LightGBMRanker and WideAndDeepRanker.
    @Override
    public DataResult process(ServiceRequest request, List<DataResult> items, DataContext context) {
        List<String> itemIds = getListData(items).stream().map(x->getField(x, itemIdCol, "")).filter(StringUtils::isNotEmpty).collect(Collectors.toList());
        List<FeatureTable> featureTables = constructNpsFeatureTables(itemIds, context);
        List<Float> scores = getVectorsFromNps(modelName, featureTables);

        Iterator<String> itemModelIt = itemIds.iterator();
        Iterator<Float> scoreIt = scores.iterator();
        List<Map> data = Lists.newArrayList();
        while (itemModelIt.hasNext() && scoreIt.hasNext()) {
            Map<String, Object> item = Maps.newHashMap();
            if (setFieldFail(item, 0, itemModelIt.next())) {
                continue;
            }
            Float score = scoreIt.next();
            Map<String, Object> value = Maps.newHashMap();
            value.put(ALGO_NAME, score);
            if (setFieldFail(item, 1, value)) {
                continue;
            }
            data.add(item);
        }
        DataResult result = new DataResult();
        result.setData(data);
        return result;
    }

    private List<Float> getVectorsFromNps(String modelName, List<FeatureTable> featureTables){
        Map<String, ArrowTensor> npsResultMap = null;
        try {
            npsResultMap = ServingClient.predictBlocking(client, modelName, featureTables, Collections.emptyMap());
        } catch (IOException e) {
            log.error("TwoTower request nps fail!");
            throw new RuntimeException(e);
        }
        return Utils.getScoresFromNpsResult(npsResultMap, TARGET_KEY, TARGET_INDEX);
    }

    private List<FeatureTable> constructNpsFeatureTables(List<String> itemIds, DataContext context) {
        ServiceRequest req = new ServiceRequest(lightGBMFeatureTask, service.getName());
        req.putIn(itemIdCol, itemIds);
        FeatureTable lightGBMTable = service.execute(lightGBMFeatureTask, req, context).getFeatureTable();
        return Lists.newArrayList(lightGBMTable);
    }
}