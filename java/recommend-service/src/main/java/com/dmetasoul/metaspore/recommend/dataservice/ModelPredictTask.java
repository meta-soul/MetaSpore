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
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.dmetasoul.metaspore.serving.PredictGrpc;
import com.dmetasoul.metaspore.serving.ServingClient;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.util.*;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation
public abstract class ModelPredictTask extends AlgoTransform {
    public static final String DEFAULT_MODEL_NAME = "two_towers_simplex";
    public static final String TARGET_KEY = "output";
    public static final int TARGET_INDEX = -1;

    private String modelName;
    private String targetKey;
    private int targetIndex;
    private String address;
    private String host;
    private int port;

    private PredictGrpc.PredictBlockingStub client;

    @Override
    public boolean initService() {
        if (!super.initService()) {
            return false;
        }
        modelName = getOptionOrDefault("modelName", DEFAULT_MODEL_NAME);
        targetKey = getOptionOrDefault("targetKey", TARGET_KEY);
        targetIndex = getOptionOrDefault("targetIndex", TARGET_INDEX);
        host = getOptionOrDefault("host", "127.0.0.1");
        port = getOptionOrDefault("port", 9091);
        ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
        client = PredictGrpc.newBlockingStub(channel);
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        List<DataResult> algoInferenceResult = getDataResultByRelys(config.getDepend().getWhen(), config.getDepend().isAny(), context);
        List<FeatureTable> featureTables = Lists.newArrayList();
        for (DataResult dataResult : algoInferenceResult) {
            if (dataResult.getFeatureTable() != null) {
                featureTables.add(dataResult.getFeatureTable());
            }
        }
        if (featureTables.isEmpty()) {
            log.error("lastChainTasks has no algoInference!");
            throw new RuntimeException("lastChainTasks has no algoInference at ModelPredictTask!");
        }
        Map<String, ArrowTensor> npsResultMap = null;
        try {
            npsResultMap = ServingClient.predictBlocking(client, modelName, featureTables, Collections.emptyMap());
        } catch (IOException e) {
            log.error("TwoTower request nps fail!");
            throw new RuntimeException(e);
        }
        DataResult.PredictResult predictResult = new DataResult.PredictResult();
        if (targetIndex < 0) {
            predictResult.setEmbedding(Utils.getVectorsFromNpsResult(npsResultMap, TARGET_KEY));
        } else {
            predictResult.setScore(Utils.getScoresFromNpsResult(npsResultMap, TARGET_KEY, TARGET_INDEX));
        }
        DataResult result = new DataResult();
        result.setPredictResult(predictResult);
        return result;
    }
}
