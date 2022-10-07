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
import com.dmetasoul.metaspore.recommend.common.ConvTools;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.*;
import com.dmetasoul.metaspore.recommend.data.*;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.serving.*;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.Assert;
import org.springframework.util.StopWatch;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.stream.Collectors;

@Data
@Slf4j
@ServiceAnnotation("AlgoTransform")
public class AlgoTransformTask extends DataService {
    protected ExecutorService taskPool;
    protected FeatureConfig.AlgoTransform algoTransform;
    protected Map<String, Function> additionFunctions;
    protected TableData fieldTableData;
    protected Map<String, String> actionTypes;

    public <T> T getOptionOrDefault(String key, T value) {
        return CommonUtils.getField(algoTransform.getOptions(), key, value);
    }

    @Override
    public boolean initService() {
        algoTransform = taskFlowConfig.getAlgoTransforms().get(name);
        this.taskPool = taskServiceRegister.getTaskPool();
        for (String col : algoTransform.getColumnNames()) {
            resFields.add(algoTransform.getFieldMap().get(col));
            dataTypes.add(algoTransform.getColumnMap().get(col));
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
        initFunctions();
        addFunctions();
        fieldTableData = new TableData();
        actionTypes = Maps.newHashMap();
        return initTask();
    }

    public boolean initTask() {
        return true;
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        return super.makeRequest(depend, null, context);
    }

    @Override
    protected void preCondition(ServiceRequest request, DataContext context) {
        super.preCondition(request, context);
        actionTypes.clear();
    }

    public void initFunctions() {
        addFunction("setValue", (fieldTableData, options, taskPool) -> {
            Object object = CommonUtils.getObject(options.getOptions(), "value");
            if (CollectionUtils.isNotEmpty(options.getTypes())) {
                DataTypeEnum dataType = ColumnInfo.getType(options.getTypes().get(0));
                Class<?> cls = dataType.getCls();
                Object value = ConvTools.parseObject(object, cls);
                if (value != null) {
                    Assert.isInstanceOf(cls, value, "setValue config value type wrong");
                }
                fieldTableData.addValue(new FieldInfo(options.getNames().get(0)), value);
            }
            return true;
        });
    }

    public void addFunctions() {
    }

    public void addFunction(String name, Function function) {
        additionFunctions.put(name, function);
    }

    public void addDataResults(List<String> names, DataContext context) {
        if (CollectionUtils.isNotEmpty(names)) {
            for (String taskName : names) {
                DataResult result = getDataResultByName(taskName, context);
                if (result != null) {
                    fieldTableData.addDataResult(taskName, result);
                }
            }
        }
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        addDataResults(algoTransform.getFeature(), context);
        addDataResults(algoTransform.getAlgoTransform(), context);
        StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
        for (FieldAction fieldAction : algoTransform.getActionList()) {
            if (StringUtils.isEmpty(fieldAction.getFunc())) {
                if (CollectionUtils.isNotEmpty(fieldAction.getNames())) {
                    int fieldSize = 0;
                    if (CollectionUtils.isNotEmpty(fieldAction.getFields())) {
                        fieldSize = fieldAction.getFields().size();
                    }
                    for (int i = 0; i < fieldAction.getNames().size(); ++i) {
                        if (i < fieldSize) {
                            fieldTableData.copyField(fieldAction.getFields().get(i), fieldAction.getNames().get(i));
                        } else if (CollectionUtils.isNotEmpty(fieldAction.getInput()) && i < fieldSize + fieldAction.getInput().size()) {
                            fieldTableData.copyField(fieldAction.getInput().get(i-fieldSize), fieldAction.getNames().get(i));
                        }
                    }
                }
            } else {
                Function function = additionFunctions.get(fieldAction.getFunc());
                if (function == null) {
                    function = taskServiceRegister.getFunction(fieldAction.getFunc());
                    if (function == null) {
                        throw new RuntimeException("function get fail at " + fieldAction.getFunc());
                    }
                }
                try {
                    timeRecorder.start(String.format("%s_fieldAction_func_%s", name, fieldAction.getFunc()));
                    fieldTableData.addFieldList(fieldAction.getNames(), fieldAction.getTypes());
                    if (!function.process(fieldTableData, fieldAction, taskPool)) {
                        throw new RuntimeException("the function process fail. func:" + fieldAction.getFunc());
                    }
                } finally {
                    timeRecorder.stop();
                }
            }
        }
        context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
        return fieldTableData.getDataResult(name, resFields);
    }

    public void setFieldData(FeatureTable featureTable, String col, DataTypeEnum dataType, List<Object> data) {
        if (!dataType.set(featureTable, col, data)) {
            log.error("set featureTable fail!");
        }
    }

    public FeatureTable convFeatureTable(String name, List<String> columns, TableData fieldTableData) {
        List<Field> inferenceFields = columns.stream().map(fieldTableData::getField).collect(Collectors.toList());
        return fieldTableData.getFeatureTable(name, inferenceFields);
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
