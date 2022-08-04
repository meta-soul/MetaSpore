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
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.TensorResult;
import com.dmetasoul.metaspore.serving.*;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation
public class AlgoInferenceTask extends AlgoTransformTask {
    private static final String DEFAULT_MODEL_NAME = "two_towers_simplex";
    private static final String TARGET_KEY = "output";
    private static final int TARGET_INDEX = -1;
    private String modelName;
    private String targetKey;
    private int targetIndex;
    private String address;
    private String host;
    private int port;

    private ManagedChannel channel;

    private PredictGrpc.PredictBlockingStub client;

    public boolean initTask() {
        modelName = getOptionOrDefault("modelName", DEFAULT_MODEL_NAME);
        targetKey = getOptionOrDefault("targetKey", TARGET_KEY);
        targetIndex = getOptionOrDefault("targetIndex", TARGET_INDEX);
        host = getOptionOrDefault("host", "127.0.0.1");
        port = getOptionOrDefault("port", 9091);
        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
        client = PredictGrpc.newBlockingStub(channel);
        return true;
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

    protected DataResult transform(FeatureTable featureTable, DataContext context) {
        TensorResult dataResult = new TensorResult();
        Map<String, ArrowTensor> npsResultMap;
        try {
            npsResultMap = ServingClient.predictBlocking(client, modelName, List.of(featureTable), Collections.emptyMap());
        } catch (IOException e) {
            log.error("TwoTower request nps fail!");
            throw new RuntimeException(e);
        }
        dataResult.setTensor(npsResultMap.get(targetKey));
        dataResult.setIndex(targetIndex);
        dataResult.setFeatureTable(featureTable);
        return dataResult;
    }
}
