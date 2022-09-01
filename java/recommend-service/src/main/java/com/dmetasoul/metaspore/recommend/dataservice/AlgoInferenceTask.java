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
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.configure.FieldAction;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.data.IndexData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.serving.*;
import com.google.common.collect.Lists;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.channel.epoll.EpollEventLoopGroup;
import io.grpc.netty.shaded.io.netty.channel.epoll.EpollSocketChannel;
import lombok.Data;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.bouncycastle.util.Strings;
import org.springframework.util.Assert;

import javax.validation.constraints.NotEmpty;
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
    protected int maxReservation;
    protected String algoName;
    protected ManagedChannel channel;
    protected PredictGrpc.PredictBlockingStub client;

    public boolean initTask() {
        modelName = getOptionOrDefault("modelName", DEFAULT_MODEL_NAME);
        targetKey = getOptionOrDefault("targetKey", TARGET_KEY);
        targetIndex = getOptionOrDefault("targetIndex", TARGET_INDEX);
        log.info("initManagedChannel name: {} algoTransform: {}", name, algoTransform);
        log.info("initManagedChannel name: {} option: {}", name, algoTransform.getOptions());
        channel = initManagedChannel(algoTransform.getOptions());
        client = PredictGrpc.newBlockingStub(channel);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        algoName = getOptionOrDefault("algo-name", "two_tower");
        return true;
    }

    public ManagedChannel initManagedChannel(Map<String, Object> option) {
        String host = Utils.getField(option, "host", "127.0.0.1");
        int port = Utils.getField(option, "port", 50000);
        NegotiationType negotiationType = NegotiationType.valueOf(Strings.toUpperCase((String) option.getOrDefault("negotiationType", "plaintext")));
        NettyChannelBuilder channelBuilder = NettyChannelBuilder.forAddress(host, port)
                .keepAliveWithoutCalls((Boolean) option.getOrDefault("enableKeepAliveWithoutCalls", false))
                .negotiationType(negotiationType)
                .keepAliveTime((Long) option.getOrDefault("keepAliveTime", 300L), TimeUnit.SECONDS)
                .keepAliveTimeout((Long) option.getOrDefault("keepAliveTimeout", 10L), TimeUnit.SECONDS);
        return channelBuilder.build();
    }

    @Override
    public void addFunctions() {
        addFunction("genEmbedding", new Function() {
            @Override
            public boolean process(@NotEmpty List<FieldData> fields, @NotEmpty List<FieldData> result, @NonNull FieldAction fieldAction) {
                Assert.isTrue(CollectionUtils.isNotEmpty(fieldAction.getAlgoColumns()), "AlgoColumns must not empty");
                List<FeatureTable> featureTables = Lists.newArrayList();
                for (Map<String, List<String>> item : fieldAction.getAlgoColumns()) {
                    for (Map.Entry<String, List<String>> entry : item.entrySet()) {
                        if (CollectionUtils.isEmpty(entry.getValue())) continue;
                        List<String> columns = Lists.newArrayList();
                        for (String name: entry.getValue()) {
                            if (MapUtils.isNotEmpty(fieldAction.getAlgoFields()) && fieldAction.getAlgoFields().containsKey(name)) {
                                FeatureConfig.Field field = fieldAction.getAlgoFields().get(name);
                                columns.add(field.toString());
                            } else {
                                columns.add(name);
                            }
                        }
                        featureTables.add(convFeatureTable(entry.getKey(), columns, fields));
                    }
                }
                String targetName = Utils.getField(fieldAction.getOptions(), "targetKey", targetKey);
                String model = Utils.getField(fieldAction.getOptions(), "modelName", modelName);
                ArrowTensor arrowTensor = predict(featureTables, model, targetName);
                List<Object> res = Lists.newArrayList();
                res.addAll(getFromTensor(arrowTensor));
                result.get(0).setValue(res, getFieldIndex(fields));
                return true;
            }
            @Override
            public boolean process(@NotEmpty List<FieldData> fields, @NotEmpty List<FieldData> result, Map<String, Object> options) {
                return false;
            }
        });
        addFunction("predictScore", new Function() {
            @Override
            public boolean process(@NotEmpty List<FieldData> fields, @NotEmpty List<FieldData> result, @NonNull FieldAction fieldAction) {
                Assert.isTrue(CollectionUtils.isNotEmpty(fieldAction.getAlgoColumns()), "AlgoColumns must not empty");
                List<FeatureTable> featureTables = Lists.newArrayList();
                for (Map<String, List<String>> item : fieldAction.getAlgoColumns()) {
                    for (Map.Entry<String, List<String>> entry : item.entrySet()) {
                        if (CollectionUtils.isEmpty(entry.getValue())) continue;
                        List<String> columns = Lists.newArrayList();
                        for (String name: entry.getValue()) {
                            if (MapUtils.isNotEmpty(fieldAction.getAlgoFields()) && fieldAction.getAlgoFields().containsKey(name)) {
                                FeatureConfig.Field field = fieldAction.getAlgoFields().get(name);
                                columns.add(field.toString());
                            } else {
                                columns.add(name);
                            }
                        }
                        featureTables.add(convFeatureTable(entry.getKey(), columns, fields));
                    }
                }
                String targetName = Utils.getField(fieldAction.getOptions(), "targetKey", targetKey);
                int index = Utils.getField(fieldAction.getOptions(), "targetIndex", targetIndex);
                String model = Utils.getField(fieldAction.getOptions(), "modelName", modelName);
                ArrowTensor arrowTensor = predict(featureTables, model, targetName);
                List<Object> res = Lists.newArrayList();
                res.addAll(getFromTensor(arrowTensor, index));
                result.get(0).setValue(res, getFieldIndex(fields));
                return true;
            }
            @Override
            public boolean process(@NotEmpty List<FieldData> fields, @NotEmpty List<FieldData> result, Map<String, Object> options) {
                return false;
            }
        });
        addFunction("rankCollectItem", (fields, result, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && fields.size() == 2,
                    "input fields must not empty");
            Assert.isTrue(fields.get(0).isMatch(DataTypeEnum.LIST_STR),
                    "rankCollectItem input[0] is itemId list<string>");
            Assert.isTrue(fields.get(1).isMatch(DataTypeEnum.LIST_DOUBLE),
                    "rankCollectItem input[1] is score list<double>");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            int limit = Utils.getField(options, "maxReservation", maxReservation);
            List<IndexData> itemIds = fields.get(0).getIndexValue();
            List<Double> scores = fields.get(1).getValue();
            for (int i = 0; i < itemIds.size() && i < limit; ++i) {
                result.get(0).addIndexData(itemIds.get(i));
                result.get(1).addIndexData(FieldData.create(itemIds.get(i).getIndex(), Map.of(algoName, Utils.get(scores, i, 0.0))));
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

    protected ArrowTensor predict(List<FeatureTable> featureTables, String modelName, String targetKey) {
        Map<String, ArrowTensor> npsResultMap;
        try {
            npsResultMap = ServingClient.predictBlocking(client, modelName, featureTables, Collections.emptyMap());
        } catch (IOException e) {
            log.error("TwoTower request nps fail!");
            throw new RuntimeException(e);
        }
        return npsResultMap.get(targetKey);
    }
}
