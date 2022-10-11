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
import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.configure.ColumnInfo;
import com.dmetasoul.metaspore.recommend.configure.FieldInfo;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.serving.*;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.bouncycastle.util.Strings;
import org.springframework.util.Assert;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Data
@Slf4j
@ServiceAnnotation("AlgoInference")
public class AlgoInferenceTask extends AlgoTransformTask {
    protected static final String DEFAULT_MODEL_NAME = "two_towers_simplex";
    protected static final String TARGET_KEY = "output";
    public static final int DEFAULT_MAX_RESERVATION = 1000;
    protected static final int TARGET_INDEX = -1;
    protected String modelName;
    protected String targetKey;
    protected int targetIndex;
    protected int maxReservation;
    protected String algoName;
    protected ManagedChannel channel;
    protected PredictGrpc.PredictBlockingStub client;

    public boolean initTask() {
        modelName = getOptionOrDefault("modelName", DEFAULT_MODEL_NAME);
        targetKey = getOptionOrDefault("targetKey", TARGET_KEY);
        targetIndex = getOptionOrDefault("targetIndex", TARGET_INDEX);
        channel = initManagedChannel(algoTransform.getOptions());
        client = PredictGrpc.newBlockingStub(channel);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        algoName = getOptionOrDefault("algo-name", "two_tower");
        return true;
    }

    public ManagedChannel initManagedChannel(Map<String, Object> option) {
        String host = CommonUtils.getField(option, "host", "127.0.0.1");
        int port = CommonUtils.getField(option, "port", 50000, Integer.class);
        NegotiationType negotiationType = NegotiationType.valueOf(Strings.toUpperCase((String) option.getOrDefault("negotiationType", "plaintext")));
        NettyChannelBuilder channelBuilder = NettyChannelBuilder.forAddress(host, port)
                .keepAliveWithoutCalls((Boolean) option.getOrDefault("enableKeepAliveWithoutCalls", false))
                .negotiationType(negotiationType)
                .keepAliveTime((Long) option.getOrDefault("keepAliveTime", 300L), TimeUnit.SECONDS)
                .keepAliveTimeout((Long) option.getOrDefault("keepAliveTimeout", 10L), TimeUnit.SECONDS);
        return channelBuilder.build();
    }

    @SuppressWarnings("unchecked")
    @Override
    public void addFunctions() {
        addFunction("genEmbedding", (fieldTableData, fieldAction, taskPool) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fieldAction.getAlgoColumns()), "AlgoColumns must not empty");
            List<FeatureTable> featureTables = Lists.newArrayList();
            for (Map<String, List<String>> item : fieldAction.getAlgoColumns()) {
                for (Map.Entry<String, List<String>> entry : item.entrySet()) {
                    if (CollectionUtils.isEmpty(entry.getValue())) continue;
                    List<String> columns = Lists.newArrayList();
                    for (String name : entry.getValue()) {
                        if (MapUtils.isNotEmpty(fieldAction.getAlgoFields()) && fieldAction.getAlgoFields().containsKey(name)) {
                            FieldInfo field = fieldAction.getAlgoFields().get(name);
                            columns.add(field.toString());
                        } else {
                            columns.add(name);
                        }
                    }
                    FeatureTable featureTable = convFeatureTable(entry.getKey(), columns, fieldTableData);
                    if (featureTable.getRowCount() == 0) {
                        log.error("model input is empty! at fieldAction: {}", fieldAction);
                        return true;
                    }
                    featureTables.add(featureTable);
                }
            }
            String targetName = CommonUtils.getField(fieldAction.getOptions(), "targetKey", targetKey);
            String model = CommonUtils.getField(fieldAction.getOptions(), "modelName", modelName);
            try (ArrowAllocator allocator = new ArrowAllocator(Integer.MAX_VALUE)) {
                ArrowTensor arrowTensor = predict(featureTables, allocator, model, targetName);
                List<Object> res = Lists.newArrayList();
                res.addAll(getFromTensor(arrowTensor));
                fieldTableData.addValueList(fieldAction.getNames().get(0), res);
                for (FeatureTable featureTable : featureTables) {
                    featureTable.close();
                }
                return true;
            }
        });
        addFunction("predictScore", (fieldTableData, fieldAction, taskPool) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fieldAction.getAlgoColumns()), "AlgoColumns must not empty");
            List<FeatureTable> featureTables = Lists.newArrayList();
            for (Map<String, List<String>> item : fieldAction.getAlgoColumns()) {
                for (Map.Entry<String, List<String>> entry : item.entrySet()) {
                    if (CollectionUtils.isEmpty(entry.getValue())) continue;
                    List<String> columns = Lists.newArrayList();
                    for (String name : entry.getValue()) {
                        if (MapUtils.isNotEmpty(fieldAction.getAlgoFields()) && fieldAction.getAlgoFields().containsKey(name)) {
                            FieldInfo field = fieldAction.getAlgoFields().get(name);
                            columns.add(field.toString());
                        } else {
                            columns.add(name);
                        }
                    }
                    FeatureTable featureTable = convFeatureTable(entry.getKey(), columns, fieldTableData);
                    if (featureTable.getRowCount() == 0) {
                        log.error("model input is empty! at fieldAction: {}", fieldAction);
                        fieldTableData.addValueList(fieldAction.getNames().get(0), List.of());
                        return true;
                    }
                    featureTables.add(featureTable);
                }
            }
            String targetName = CommonUtils.getField(fieldAction.getOptions(), "targetKey", targetKey);
            int index = CommonUtils.getField(fieldAction.getOptions(), "targetIndex", targetIndex);
            String model = CommonUtils.getField(fieldAction.getOptions(), "modelName", modelName);
            try (ArrowAllocator allocator = new ArrowAllocator(Integer.MAX_VALUE)) {
                ArrowTensor arrowTensor = predict(featureTables, allocator, model, targetName);
                List<Object> res = Lists.newArrayList();
                res.addAll(getFromTensor(arrowTensor, index));
                fieldTableData.addValueList(fieldAction.getNames().get(0), res);
                for (FeatureTable featureTable : featureTables) {
                    featureTable.close();
                }
                return true;
            }
        });
        addFunction("rankCollectItem", (fieldTableData, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(CollectionUtils.isNotEmpty(config.getInputFields()) && config.getInputFields().size() > 1,
                    "input must has >= 2 field");
            int limit = CommonUtils.getField(options, "maxReservation", maxReservation);
            FieldInfo itemid = config.getInputFields().get(1);
            FieldInfo scores = config.getInputFields().get(2);
            FieldInfo originScores = config.getInputFields().get(0);
            List<String> names = config.getNames();
            List<Object> types = config.getTypes();
            for (int i = 0; i < fieldTableData.getData().size() && i < limit; ++i) {
                if (names.size() > 0) {
                    fieldTableData.setValue(i, names.get(0), fieldTableData.getValue(i, itemid));
                }
                float scoreValue = (Float) fieldTableData.getValue(i, scores, 0.0F);
                if (names.size() > 1) {
                    fieldTableData.setValue(i, names.get(1), scoreValue);
                }
                if (names.size() > 2) {
                    Map<String, Double> originScoreValue = Maps.newHashMap();
                    originScoreValue.put(algoName, (double) scoreValue);
                    if (originScores != null && fieldTableData.getDataSchema().containsKey(originScores)
                            && DataTypeEnum.MAP_STR_DOUBLE.equals(fieldTableData.getType(originScores))) {
                        originScoreValue.putAll(
                                (Map<? extends String, ? extends Double>) fieldTableData.getValue(i,
                                        originScores, Maps.newHashMap()));
                    }
                    fieldTableData.setValue(i, names.get(2), originScoreValue);
                }
            }
            return true;
        });
    }

    @SneakyThrows
    @Override
    public void close() {
        if (channel == null || channel.isShutdown()) return;
        channel.shutdown().awaitTermination(1, TimeUnit.SECONDS);
    }

    protected ArrowTensor predict(List<FeatureTable> featureTables, ArrowAllocator allocator,
                                  String modelName, String targetKey) {
        Map<String, ArrowTensor> npsResultMap;
        try {
            npsResultMap = ServingClient.predictBlocking(client, modelName,
                    featureTables, allocator, Collections.emptyMap());
        } catch (IOException e) {
            log.error("TwoTower request nps fail!");
            throw new RuntimeException(e);
        }
        return npsResultMap.get(targetKey);
    }
}
