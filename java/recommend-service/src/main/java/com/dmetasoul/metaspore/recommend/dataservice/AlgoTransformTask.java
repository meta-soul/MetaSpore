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
import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.AggregateFunction;
import com.dmetasoul.metaspore.recommend.functions.FlatFunction;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.recommend.functions.ScatterFunction;
import com.dmetasoul.metaspore.serving.*;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.Assert;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.stream.Collectors;

@Data
@Slf4j
@ServiceAnnotation("AlgoTransform")
public class AlgoTransformTask extends DataService {
    protected ExecutorService taskPool;
    protected FeatureConfig.AlgoTransform algoTransform;
    protected Map<String, Function> functionMap;
    protected Map<String, Function> additionFunctions;
    protected Map<String, List<Object>> actionResult;
    protected Map<String, String> actionTypes;
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
        depend = new Chain();
        List<String> depends = Lists.newArrayList();
        if (CollectionUtils.isNotEmpty(algoTransform.getFeature())) {
            depends.addAll(algoTransform.getFeature());
        }
        if (CollectionUtils.isNotEmpty(algoTransform.getAlgoTransform())) {
            depends.addAll(algoTransform.getAlgoTransform());
        }
        depend.setWhen(depends);
        depend.setAny(false);
        additionFunctions = Maps.newHashMap();
        addFunctions();
        actionResult = Maps.newHashMap();
        actionTypes = Maps.newHashMap();
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
        actionTypes.clear();
        actionIndex.clear();
    }

    public void initFunctions() {
        addFunction("flatList", (FlatFunction) (indexs, fields, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && indexs != null, "input data is not null");
            List<Object> res = Lists.newArrayList();
            List<Object> input = fields.get(0).getValue();
            int num = 0;
            for (int i = 0; i < input.size(); ++i) {
                Object item = input.get(i);
                Assert.isInstanceOf(Collection.class, item);
                Collection<?> list = (Collection<?>) item;
                for (Object o : list) {
                    num += 1;
                    indexs.add(i);
                    res.add(o);
                }
            }
            return res;
        });
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
            actionResult.clear();
            actionIndex.clear();
            for (int i = 0; i < res.size(); ++i) {
                actionIndex.add(i);
            }
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
        for (int i = maxIndex + 1; i < res.size(); ++i) {
            data.add(res.get(i));
            newIndex.add(i);
        }
        if (needNewResult) {
            for (String key: actionResult.keySet()) {
                List<Object> oldList = actionResult.get(key);
                List<Object> newList = Lists.newArrayList();
                for (int i : indexResult) {
                    newList.add(get(oldList,i));
                }
                actionResult.put(key, newList);
            }
        }
        actionIndex = newIndex;
        actionResult.put(col, data);
    }

    private Object get(List<Object> list, int index) {
        if (CollectionUtils.isNotEmpty(list) && index >= 0 && index < list.size()) {
            return list.get(index);
        }
        return null;
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
            data.add(get(res,i));
        }
        for (int i = maxIndex + 1; i < res.size(); ++i) {
            data.add(res.get(i));
            actionIndex.add(i);
        }
        actionResult.put(col, data);
    }

    private void alignActionResult(String col, Map<String, List<Object>> res, List<String> names) {
        if (CollectionUtils.isEmpty(actionIndex)) {
            for (int i = 0; i < res.size(); ++i) {
                actionIndex.add(i);
            }
        }
        int maxIndex = 0;
        for (String key: names) {
            List<Object> oldList = res.get(key);
            List<Object> newList = Lists.newArrayList();
            for (int i : actionIndex) {
                if (i > maxIndex) maxIndex = i;
                newList.add(get(oldList,i));
            }
            for (int i = maxIndex + 1; i < oldList.size(); ++i) {
                newList.add(oldList.get(i));
                actionIndex.add(i);
            }
            actionResult.put(key, newList);
        }
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
        result.addAll(getDataResultByNames(algoTransform.getAlgoTransform(), context));
        FeatureTable featureTable = new FeatureTable(name, resFields, ArrowAllocator.getAllocator());
        Map<String, DataResult> dataResultMap = getDataResults(result);
        for (FeatureConfig.FieldAction fieldAction : algoTransform.getActionList()) {
            List<FeatureConfig.Field> fields = fieldAction.getFields();
            List<String> input = fieldAction.getInput();
            List<FieldData> fieldDatas = Lists.newArrayList();
            if (CollectionUtils.isNotEmpty(fields)) {
                for (FeatureConfig.Field field : fields) {
                    DataResult dataResult = dataResultMap.get(field.getTable());
                    List<Object> itemData = dataResult.get(field.getFieldName());
                    alignActionResult(field.toString(), itemData);
                    FeatureConfig.Feature feature = taskFlowConfig.getFeatures().get(field.getTable());
                    fieldDatas.add(FieldData.of(field.toString(),
                            DataTypes.getDataType(feature.getColumnMap().get(field.getFieldName())), actionResult.get(field.toString())));
                }
            }
            if (CollectionUtils.isNotEmpty(input)) {
                for (String item : input) {
                    List<Object> itemData = actionResult.get(item);
                    Assert.notNull(itemData, "use flat function result, no found result at input: " + item);
                    fieldDatas.add(FieldData.of(item, DataTypes.getDataType(actionTypes.get(item)), itemData));
                }
            }
            List<Object> res = fieldDatas.get(0).getValue();
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
                    res = flatFunction.flat(indexs, fieldDatas, fieldAction.getOptions());
                    Assert.notNull(res, "process result is null at: " + fieldAction.getName());
                    alignActionResult(indexs, fieldAction.getName(), res);
                    actionTypes.put(fieldAction.getName(), fieldAction.getType());
                } else if (function instanceof AggregateFunction) {
                    AggregateFunction aggregateFunction = (AggregateFunction) function;
                    Object obj = aggregateFunction.aggregate(fieldDatas, fieldAction.getOptions());
                    Assert.notNull(res, "process result is null at: " + fieldAction.getName());
                    alignActionResult(fieldAction.getName(), obj);
                    actionTypes.put(fieldAction.getName(), fieldAction.getType());
                } else if (function instanceof ScatterFunction) {
                    ScatterFunction scatterFunction = (ScatterFunction) function;
                    FeatureConfig.ScatterFieldAction scatterAction = (FeatureConfig.ScatterFieldAction) fieldAction;
                    Map<String, List<Object>> scatterRes = scatterFunction.scatter(fieldDatas, scatterAction.getNames(), fieldAction.getOptions());
                    Assert.notNull(scatterRes, "process result is null at: " + fieldAction.getName());
                    alignActionResult(fieldAction.getName(), scatterRes, scatterAction.getNames());
                    for (int i = 0; i < scatterAction.getNames().size(); ++i) {
                        actionTypes.put(scatterAction.getNames().get(i), scatterAction.getTypes().get(i));
                    }
                } else {
                    res = function.process(fieldDatas, fieldAction.getOptions());
                    Assert.notNull(res, "process result is null at: " + fieldAction.getName());
                    alignActionResult(fieldAction.getName(), res);
                    actionTypes.put(fieldAction.getName(), fieldAction.getType());
                }
            } else {
                alignActionResult(fieldAction.getName(), res);
                actionTypes.put(fieldAction.getName(), fieldAction.getType());
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
            log.error("set featureTable fail!");
        }
    }

    public FeatureTable convFeatureTable(String name, List<FieldData> fields) {
        List<Field> inferenceFields = fields.stream().map(x->Field.nullable(x.getName(), x.getType().getType()))
                .collect(Collectors.toList());
        FeatureTable featureTable = new FeatureTable(name, inferenceFields, ArrowAllocator.getAllocator());
        for (FieldData fieldData: fields) {
            if (!fieldData.getType().set(featureTable, fieldData.getName(), fieldData.getValue())) {
                log.error("set featureTable fail!");
            }
        }
        return featureTable;
    }
    public <T> List<List<T>> getFromTensor(ArrowTensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("tensor or shape is null");
        }
        ArrowTensor.TensorAccessor<T> accessor = getTensorAccessor(tensor);
        long[] shape = tensor.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Shape length must equal to 2 (batch, vector dim). shape.length: " + shape.length);
        }
        List<List<T>> vectors = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            List<T> vector = new ArrayList<>();
            for (int j = 0; j < shape[1]; j++) {
                vector.add(accessor.get(i, j));
            }
            vectors.add(vector);
        }
        return vectors;
    }

    public <T> List<T> getFromTensor(ArrowTensor tensor, int targetIndex) {
        if (tensor == null) {
            throw new IllegalArgumentException("tensor or shape is null");
        }
        ArrowTensor.TensorAccessor<T> accessor = getTensorAccessor(tensor);
        long[] shape = tensor.getShape();
        if (targetIndex < 0 || targetIndex >= shape.length) {
            throw new IllegalArgumentException("Target index is out of shape scope. targetIndex: " + targetIndex);
        }
        List<T> scores = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            scores.add(accessor.get(i, targetIndex));
        }
        return scores;
    }

    @SuppressWarnings("unchecked")
    private <T> ArrowTensor.TensorAccessor<T> getTensorAccessor(ArrowTensor tensor) {
        if (tensor == null) throw new IllegalArgumentException("tensor is null");
        if (tensor.isFloatTensor()) {
            return (ArrowTensor.TensorAccessor<T>) tensor.getFloatData();
        } else if (tensor.isDoubleTensor()) {
            return (ArrowTensor.TensorAccessor<T>) tensor.getDoubleData();
        } else if (tensor.isLongTensor()) {
            return (ArrowTensor.TensorAccessor<T>) tensor.getLongData();
        } else {
            return (ArrowTensor.TensorAccessor<T>) tensor.getIntData();
        }
    }
}
