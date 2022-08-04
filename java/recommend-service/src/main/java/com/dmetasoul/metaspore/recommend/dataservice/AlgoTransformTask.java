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
import com.dmetasoul.metaspore.recommend.functions.AggregateFunction;
import com.dmetasoul.metaspore.recommend.functions.FlatFunction;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.serving.*;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.springframework.util.Assert;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation
public class AlgoTransformTask extends DataService {
    protected ExecutorService taskPool;
    protected FeatureConfig.AlgoTransform algoTransform;
    protected List<FeatureConfig.Feature> features;
    protected Map<String, Function> functionMap;
    protected Map<String, Function> additionFunctions;
    protected Map<String, List<Object>> actionResult;
    private List<Integer> actionIndex;
    public <T> T getOptionOrDefault(String key, T value) {
        return Utils.getField(algoTransform.getOptions(), key, value);
    }

    @Override
    public boolean initService() {
        algoTransform = taskFlowConfig.getAlgoTransforms().get(name);
        this.taskPool = taskServiceRegister.getTaskPool();
        functionMap = taskServiceRegister.getFunctions();
        for (String col: algoTransform.getColumnNames()) {
            String type = algoTransform.getColumnMap().get(col);
            DataTypeEnum dataType = DataTypes.getDataType(type);
            resFields.add(Field.nullable(col, dataType.getType()));
            dataTypes.add(dataType);
        }
        features = Lists.newArrayList();
        for (String feature : algoTransform.getFeature()) {
            features.add(taskFlowConfig.getFeatures().get(feature));
        }
        depend = new Chain();
        List<String> depends = algoTransform.getFeature();
        depend.setThen(depends);
        additionFunctions = Maps.newHashMap();
        addFunctions();
        actionResult = Maps.newHashMap();
        actionIndex = Lists.newArrayList();
        return initTask();
    }

    public boolean initTask() {
        return true;
    }

    @Override
    protected void preCondition(ServiceRequest request, DataContext context) {
        super.preCondition(request, context);
        actionResult.clear();
        actionIndex.clear();
    }

    public void addFunctions() {}

    public void addFunction(String name, Function function) {
        additionFunctions.put(name, function);
    }
    protected Map<String, DataResult> getDataResults(List<DataResult> result) {
        Map<String, DataResult> data = Maps.newHashMap();
        if (CollectionUtils.isNotEmpty(result)) {
            for (DataResult dataResult : result) {
                data.put(dataResult.getName(), dataResult);
            }
        }
        return data;
    }

    private void alignActionResult(List<Integer> indexs, String col, List<Object> res) {
        if (CollectionUtils.isEmpty(indexs)) {
            actionResult.put(col, res);
            return;
        }
        if (CollectionUtils.isEmpty(actionIndex)) {
            actionIndex = indexs;
            actionResult.put(col, res);
            return;
        }
        List<Integer> newIndex = Lists.newArrayList();
        List<Object> data = Lists.newArrayList();
        Map<Integer, List<Integer>> inputIndex = Maps.newHashMap();
        for (int i = 0; i < indexs.size(); ++i) {
            inputIndex.computeIfAbsent(indexs.get(i), k->Lists.newArrayList()).add(i);
        }
        List<Integer> indexResult = Lists.newArrayList();
        boolean needNewResult = false;
        int maxIndex = 0;
        for (int i = 0; i < actionIndex.size(); ++i) {
            int index = actionIndex.get(i);
            List<Integer> list = inputIndex.get(index);
            if (CollectionUtils.isEmpty(list)) {
                indexResult.add(i);
                newIndex.add(index);
                data.add(null);
            } else {
                needNewResult = true;
                for (int k : list) {
                    indexResult.add(i);
                    newIndex.add(index);
                    data.add(res.get(k));
                    if (k > maxIndex) maxIndex = k;
                }
            }
        }
        for (int i = maxIndex + 1; i < data.size(); ++i) {
            data.add(res.get(i));
            newIndex.add(i);
        }
        if (needNewResult) {
            for (String key: actionResult.keySet()) {
                List<Object> oldList = actionResult.get(key);
                List<Object> newList = Lists.newArrayList();
                for (int i : indexResult) {
                    newList.add(oldList.get(i));
                }
                actionResult.put(key, newList);
            }
        }
        actionIndex = newIndex;
        actionResult.put(col, data);
    }

    private void alignActionResult(String col, List<Object> res) {
        if (CollectionUtils.isEmpty(actionIndex)) {
            for (int i = 0; i < res.size(); ++i) {
                actionIndex.add(i);
            }
        }
        List<Object> data = Lists.newArrayList();
        int maxIndex = 0;
        for (int i : actionIndex) {
            if (i > maxIndex) maxIndex = i;
            data.add(res.get(i));
        }
        for (int i = maxIndex + 1; i < data.size(); ++i) {
            data.add(res.get(i));
            actionIndex.add(i);
        }
        actionResult.put(col, data);
    }

    private void alignActionResult(String col, Object res) {
        if (CollectionUtils.isEmpty(actionIndex)) {
            actionIndex.add(0);
        }
        List<Object> data = Lists.newArrayList();
        for (int i : actionIndex) {
            data.add(res);
        }
        actionResult.put(col, data);
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        List<DataResult> result = getDataResultByNames(algoTransform.getFeature(), context);
        FeatureTable featureTable = new FeatureTable(name, resFields, ArrowAllocator.getAllocator());
        Map<String, DataResult> dataResultMap = getDataResults(result);
        for (FeatureConfig.FieldAction fieldAction : algoTransform.getActionList()) {
            List<FeatureConfig.Field> fields = fieldAction.getFields();
            List<String> input = fieldAction.getInput();
            List<List<Object>> fieldValues = Lists.newArrayList();

            List<DataTypeEnum> fieldTypes = Lists.newArrayList();
            if (CollectionUtils.isNotEmpty(fields)) {
                for (FeatureConfig.Field field : fields) {
                    DataResult dataResult = dataResultMap.get(field.getTable());
                    List<Object> itemData = dataResult.get(field.getFieldName());

                    fieldValues.add(itemData);
                    fieldTypes.add(dataResult.getType(field.getFieldName()));
                }
            }
            if (CollectionUtils.isNotEmpty(input)) {
                for (String item : input) {
                    List<Object> itemData = actionResult.get(item);
                    Assert.notNull(itemData, "no found result at input: " + item);
                    fieldValues.add(itemData);
                    fieldTypes.add(DataTypes.getDataType(fieldAction.getType()));
                }
            }
            List<Object> res = fieldValues.get(0);
            if (StringUtils.isNotEmpty(fieldAction.getFunc())) {
                Function function = additionFunctions.get(fieldAction.getFunc());
                if (function == null) {
                    function = functionMap.get(fieldAction.getFunc());
                    if (function == null) {
                        throw new RuntimeException("function get fail at " + fieldAction.getFunc());
                    }
                }
                if (function instanceof FlatFunction) {
                    FlatFunction flatFunction = (FlatFunction) function;
                    List<Integer> indexs = Lists.newArrayList();
                    res = flatFunction.flat(indexs, fieldValues, fieldTypes, fieldAction.getOptions());
                    Assert.notNull(res, "process result is null at: " + fieldAction.getName());
                    alignActionResult(indexs, fieldAction.getName(), res);
                } else if (function instanceof AggregateFunction) {
                    AggregateFunction aggregateFunction = (AggregateFunction) function;
                    Object obj = aggregateFunction.aggregate(fieldValues, fieldTypes, fieldAction.getOptions());
                    Assert.notNull(res, "process result is null at: " + fieldAction.getName());
                    alignActionResult(fieldAction.getName(), obj);
                } else {
                    res = function.process(fieldValues, fieldTypes, fieldAction.getOptions());
                    Assert.notNull(res, "process result is null at: " + fieldAction.getName());
                    alignActionResult(fieldAction.getName(), res);
                }
            } else {
                alignActionResult(fieldAction.getName(), res);
            }
        }
        for (int i = 0; i < algoTransform.getColumnNames().size(); ++i) {
            String col = algoTransform.getColumnNames().get(i);
            DataTypeEnum dataType = dataTypes.get(i);
            List<Object> itemData = actionResult.get(col);
            Assert.notNull(itemData, "no found result at col: " + col);
            setFieldData(featureTable, col, dataType, itemData);
        }
        return transform(featureTable, context);
    }

    protected DataResult transform(FeatureTable featureTable, DataContext context) {
        DataResult dataResult = new DataResult();
        dataResult.setFeatureTable(featureTable);
        return dataResult;
    }

    public void setFieldData(FeatureTable featureTable, String col, DataTypeEnum dataType, List<Object> data) {
        if (!dataType.set(featureTable, col, data)) {
            log.error("set featuraTable fail!");
        }
    }
}
