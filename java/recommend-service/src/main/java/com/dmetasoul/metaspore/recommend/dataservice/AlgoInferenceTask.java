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
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.ScatterFunction;
import com.dmetasoul.metaspore.serving.*;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Data
@Slf4j
@ServiceAnnotation("AlgoInference")
public class AlgoInferenceTask extends AlgoTransformTask {
    protected static final String DEFAULT_MODEL_NAME = "two_towers_simplex";
    protected static final String TARGET_KEY = "output";
    public static final int DEFAULT_MAX_RESERVATION = 50;
    protected static final int TARGET_INDEX = -1;
    protected String modelName;
    protected String targetKey;
    protected int targetIndex;
    protected String address;
    protected String host;
    protected int port;
    protected int maxReservation;
    protected String algoName;

    protected ManagedChannel channel;

    protected PredictGrpc.PredictBlockingStub client;

    public boolean initTask() {
        modelName = getOptionOrDefault("modelName", DEFAULT_MODEL_NAME);
        targetKey = getOptionOrDefault("targetKey", TARGET_KEY);
        targetIndex = getOptionOrDefault("targetIndex", TARGET_INDEX);
        host = getOptionOrDefault("host", "127.0.0.1");
        port = getOptionOrDefault("port", 9091);
        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
        client = PredictGrpc.newBlockingStub(channel);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        algoName = getOptionOrDefault("algo-name", "itemCF");
        return true;
    }

    @Override
    public void addFunctions() {
        addFunction("genEmbedding", (fields, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields), "input fields must not empty");
            FeatureTable featureTable = convFeatureTable(String.format("embedding_%s", name), fields);
            String targetName = Utils.getField(options, "targetKey", targetKey);
            ArrowTensor arrowTensor = predict(featureTable, targetName);
            List<Object> res = Lists.newArrayList();
            res.addAll(getFromTensor(arrowTensor));
            return res;
        });
        addFunction("predictScore", (fields, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields), "input fields must not empty");
            FeatureTable featureTable = convFeatureTable(String.format("predict_%s", name), fields);
            String targetName = Utils.getField(options, "targetKey", targetKey);
            int index = Utils.getField(options, "targetIndex", targetIndex);
            ArrowTensor arrowTensor = predict(featureTable, targetName);
            List<Object> res = Lists.newArrayList();
            res.addAll(getFromTensor(arrowTensor, index));
            return res;
        });
        addFunction("rankCollectItem", (ScatterFunction) (fields, names, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && fields.size() == 2,
                    "input fields must not empty");
            Assert.isTrue(fields.get(0).isMatch(DataTypeEnum.LIST_STR),
                    "rankCollectItem input[0] is itemId list<string>");
            Assert.isTrue(fields.get(1).isMatch(DataTypeEnum.LIST_DOUBLE),
                    "rankCollectItem input[1] is score list<double>");
            int limit = Utils.getField(options, "maxReservation", maxReservation);
            List<String> itemIds = fields.get(0).getValue();
            List<Double> scores = fields.get(1).getValue();
            Map<String, List<Object>> res = Maps.newHashMap();
            for (int i = 0; i < itemIds.size() && i < limit; ++i) {
                res.computeIfAbsent(names.get(0), k->Lists.newArrayList()).add(itemIds.get(i));
                res.computeIfAbsent(names.get(1), k->Lists.newArrayList()).add(Map.of(algoName, Utils.get(scores, i, 0.0)));
            }
            return res;
        });
    }
    @Override
    public void close() {
        try {
            while(!channel.isTerminated() && channel.awaitTermination(10, TimeUnit.MILLISECONDS)) {
                Thread.yield();
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
    protected ArrowTensor predict(FeatureTable featureTable, String targetKey) {
        Map<String, ArrowTensor> npsResultMap;
        try {
            npsResultMap = ServingClient.predictBlocking(client, modelName, List.of(featureTable), Collections.emptyMap());
        } catch (IOException e) {
            log.error("TwoTower request nps fail!");
            throw new RuntimeException(e);
        }
        return npsResultMap.get(targetKey);
    }
}
