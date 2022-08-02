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
import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.serving.*;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation
public class AlgoInferenceTask extends DataService {
    private static final String DEFAULT_MODEL_NAME = "two_towers_simplex";
    private static final String TARGET_KEY = "output";
    private static final int TARGET_INDEX = -1;
    protected ExecutorService taskPool;
    protected FeatureConfig.AlgoInference algoInference;
    private List<FeatureConfig.Feature> features;
    protected Map<String, Function> functionMap;
    private List<Field> algoFields;
    private String modelName;
    private String targetKey;
    private int targetIndex;
    private String address;
    private String host;
    private int port;

    private ManagedChannel channel;

    private PredictGrpc.PredictBlockingStub client;

    public <T> T getOptionOrDefault(String key, T value) {
        return Utils.getField(algoInference.getOptions(), key, value);
    }

    @Override
    public boolean initService() {
        algoInference = taskFlowConfig.getAlgoInferences().get(name);
        this.taskPool = taskServiceRegister.getTaskPool();
        functionMap = taskServiceRegister.getFunctions();
        return initTask();
    }

    public boolean initTask() {
        algoFields = Lists.newArrayList();
        for (FeatureConfig.FieldAction fieldAction: algoInference.getFieldActions()) {
            DataTypeEnum dataType = DataTypes.getDataType(fieldAction.getType());
            algoFields.add(Field.nullable(fieldAction.getName(), dataType.getType()));
        }
        features = Lists.newArrayList();
        for (String feature : algoInference.getFeature()) {
            features.add(taskFlowConfig.getFeatures().get(feature));
        }
        Chain chain = new Chain();
        List<String> depends = algoInference.getFeature();
        chain.setThen(depends);
        taskFlow.offer(chain);
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

    private Map<String, DataResult.FeatureArray> getFeatureArrays(List<DataResult> result) {
        Map<String, DataResult.FeatureArray> featureArrays = Maps.newHashMap();
        if (CollectionUtils.isNotEmpty(result)) {
            for (DataResult dataResult : result) {
                DataResult.FeatureArray featureArray = dataResult.getFeatureArray();
                if (featureArray != null) {
                    featureArrays.put(dataResult.getName(), featureArray);
                }
            }
        }
        return featureArrays;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        List<DataResult> result = getDataResultByNames(algoInference.getFeature(), context);
        FeatureTable featureTable = new FeatureTable(name, algoFields, ArrowAllocator.getAllocator());
        Map<String, DataResult.FeatureArray> featureArrays = getFeatureArrays(result);
        for (FeatureConfig.FieldAction fieldAction : algoInference.getFieldActions()) {
            List<FeatureConfig.Field> fields = fieldAction.getFields();
            if (StringUtils.isEmpty(fieldAction.getFunc()) && CollectionUtils.isNotEmpty(fields)) {
                DataResult.FeatureArray featureArray = featureArrays.get(fields.get(0).getTable());
                setFieldData(featureTable, fieldAction.getName(), fieldAction.getType(), featureArray.getArray(fields.get(0).getFieldName()));
                continue;
            }
            List<List<Object>> fieldValues = Lists.newArrayList();
            List<String> fieldTypes = Lists.newArrayList();
            for (FeatureConfig.Field field : fieldAction.getFields()) {
                DataResult.FeatureArray featureArray = featureArrays.get(field.getTable());
                fieldValues.add(featureArray.getArray(field.getFieldName()));
                FeatureConfig.Feature feature = taskFlowConfig.getFeatures().get(field.getTable());
                fieldTypes.add(feature.getColumnMap().get(field.getFieldName()));
            }
            Function function = functionMap.get(fieldAction.getFunc());
            if (function == null) {
                log.error("function：{} get fail！", fieldAction.getFunc());
                return null;
            }
            List<Object> res = function.process(fieldValues, fieldTypes, fieldAction.getOptions());
            setFieldData(featureTable, fieldAction.getName(), fieldAction.getType(), res);
        }
        return transform(featureTable, context);
    }

    protected DataResult transform(FeatureTable featureTable, DataContext context) {
        DataResult dataResult = new DataResult();
        Map<String, ArrowTensor> npsResultMap = null;
        try {
            npsResultMap = ServingClient.predictBlocking(client, modelName, List.of(featureTable), Collections.emptyMap());
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
        dataResult.setPredictResult(predictResult);
        return dataResult;
    }

    public void setFieldData(FeatureTable featureTable, String col, String type, List<Object> data) {
        DataTypeEnum dataType = DataTypes.getDataType(type);
        if (!dataType.set(featureTable, col, data)) {
            log.error("set featuraTable fail!");
        }
    }
}
