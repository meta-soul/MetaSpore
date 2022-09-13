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
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.configure.FieldAction;
import com.dmetasoul.metaspore.recommend.configure.FieldInfo;
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

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.stream.Collectors;

import static com.dmetasoul.metaspore.recommend.configure.ColumnInfo.getField;
import static com.dmetasoul.metaspore.recommend.configure.ColumnInfo.getType;

@Data
@Slf4j
@ServiceAnnotation("AlgoTransform")
public class AlgoTransformTask extends DataService {
    protected ExecutorService taskPool;
    protected FeatureConfig.AlgoTransform algoTransform;
    protected Map<String, Function> additionFunctions;
    protected Map<String, FieldData> actionResult;
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
        actionResult = Maps.newHashMap();
        actionTypes = Maps.newHashMap();
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
    }

    public void initFunctions() {
        addFunction("flatList", (fields, result, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && CollectionUtils.isNotEmpty(result), "input and result must not null");
            List<IndexData> res = Lists.newArrayList();
            List<IndexData> input = fields.get(0).getIndexValue();
            for (IndexData item : input) {
                Assert.isInstanceOf(Collection.class, item.getVal());
                Collection<?> list = item.getVal();
                for (Object o : list) {
                    res.add(FieldData.create(item.getIndex(), o));
                }
            }
            result.get(0).setIndexValue(res);
            return true;
        });
        addFunction("multiFlatList", (fields, result, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && CollectionUtils.isNotEmpty(result), "input and result must not null");
            Assert.isTrue(fields.size() == result.size(), "input and result must be same size");
            for (int i = 0; i < fields.size(); ++i) {
                List<IndexData> res = Lists.newArrayList();
                List<IndexData> input = fields.get(i).getIndexValue();
                for (IndexData item : input) {
                    Assert.isInstanceOf(Collection.class, item.getVal());
                    Collection<?> list = item.getVal();
                    for (Object o : list) {
                        res.add(FieldData.create(item.getIndex(), o));
                    }
                }
                result.get(i).setIndexValue(res);
            }
            return true;
        });
    }

    public void addFunctions() {
    }

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

    private int getNextNum(Map<Integer, Integer> dataIndex, Map<Integer, IndexData> resItem) {
        if (MapUtils.isEmpty(dataIndex) || MapUtils.isEmpty(resItem)) return 0;
        int next = 0;
        for (int id : resItem.keySet()) {
            if (next < (dataIndex.get(id) + 1)) next = dataIndex.get(id) + 1;
        }
        return next;
    }

    private List<List<IndexData>> getIndexData(List<FieldData> fieldDatas, int index) {
        if (CollectionUtils.isEmpty(fieldDatas)) return null;
        List<List<IndexData>> res = Lists.newArrayList();
        List<Map<Integer, List<IndexData>>> dataList = Lists.newArrayList();
        Map<String, Integer> indexMap = Maps.newHashMap();
        Map<Integer, Integer> maxIndexs = Maps.newHashMap();
        Map<Integer, Integer> dataIndex = Maps.newHashMap();
        for (int i = 0; i < fieldDatas.size(); ++i) {
            FieldData itemData = fieldDatas.get(i);
            String relField = algoTransform.getColumnRel().get(itemData.getName());
            if (!indexMap.containsKey(relField)) {
                indexMap.put(relField, indexMap.size());
                dataList.add(Maps.newHashMap());
            }
            List<IndexData> data = itemData.getIndexData(index);
            int fieldNum = indexMap.get(relField);
            dataList.get(fieldNum).put(i, data);
            dataIndex.put(i, fieldNum);
            maxIndexs.put(fieldNum, Math.max(maxIndexs.getOrDefault(fieldNum, -1), data.size()));
        }
        Queue<Map<Integer, IndexData>> queue = new ArrayDeque<>();
        for (int i = 0; i < maxIndexs.get(0); ++i) {
            Map<Integer, IndexData> resItem = Maps.newHashMap();
            for (Map.Entry<Integer, List<IndexData>> entry : dataList.get(0).entrySet()) {
                resItem.put(entry.getKey(), CommonUtils.get(entry.getValue(), i, null));
            }
            queue.add(resItem);
        }
        while (!queue.isEmpty()) {
            Map<Integer, IndexData> resItem = queue.poll();
            if (MapUtils.isEmpty(resItem)) continue;
            int nextNum = getNextNum(dataIndex, resItem);
            if (nextNum >= dataList.size()) {
                List<IndexData> item = Lists.newArrayList();
                for (int i = 0; i < fieldDatas.size(); ++i) {
                    item.add(resItem.get(i));
                }
                res.add(item);
            } else {
                for (int i = 0; i < maxIndexs.get(nextNum); ++i) {
                    Map<Integer, IndexData> newItem = Maps.newHashMap();
                    newItem.putAll(resItem);
                    for (Map.Entry<Integer, List<IndexData>> entry : dataList.get(nextNum).entrySet()) {
                        newItem.put(entry.getKey(), CommonUtils.get(entry.getValue(), i, null));
                    }
                    queue.add(newItem);
                }
            }
        }
        return res;
    }

    private List<List<IndexData>> getIndexData(List<String> orderColumns, Map<String, FieldData> actionResult, int index) {
        if (CollectionUtils.isEmpty(orderColumns)) return null;
        List<FieldData> fieldDatas = Lists.newArrayList();
        for (String col : orderColumns) {
            FieldData itemData = actionResult.get(col);
            Assert.notNull(itemData, "no found result at col: " + col);
            fieldDatas.add(itemData);
        }
        return getIndexData(fieldDatas, index);
    }

    public List<Integer> getFieldIndex(List<FieldData> fieldDatas) {
        List<Integer> res = Lists.newArrayList();
        if (CollectionUtils.isEmpty(fieldDatas)) return res;
        int maxIndex = -1;
        for (FieldData itemData : fieldDatas) {
            if (maxIndex < itemData.getIndexValue().size()) maxIndex = itemData.getIndexValue().size();
        }
        for (int index = 0; index <= maxIndex; ++index) {
            for (FieldData itemData : fieldDatas) {
                IndexData item = CommonUtils.get(itemData.getIndexValue(), index, null);
                if (item == null) {
                    continue;
                }
                res.add(item.getIndex());
                break;
            }
        }
        return res;
    }

    public List<FieldData> getFieldDataList(List<FieldData> fieldDatas) {
        List<FieldData> res = Lists.newArrayList();
        if (CollectionUtils.isEmpty(fieldDatas)) return res;
        int maxIndex = -1;
        for (FieldData itemData : fieldDatas) {
            res.add(FieldData.of(itemData.getName(), itemData.getType(), itemData.getField()));
            if (maxIndex < itemData.getMaxIndex()) maxIndex = itemData.getMaxIndex();
        }
        for (int index = 0; index <= maxIndex; ++index) {
            List<List<IndexData>> data = getIndexData(fieldDatas, index);
            if (CollectionUtils.isNotEmpty(data)) {
                for (List<IndexData> list : data) {
                    for (int i = 0; i < res.size(); ++i) {
                        res.get(i).addIndexData(FieldData.create(index, list.get(i).getVal()));
                    }
                }
            }
        }
        return res;
    }

    public void setTableData(List<String> orderColumns, Map<String, FieldData> actionResult, FeatureTable table) {
        if (CollectionUtils.isEmpty(orderColumns)) return;
        int maxIndex = -1;
        for (String col : orderColumns) {
            FieldData itemData = actionResult.get(col);
            if (maxIndex < itemData.getMaxIndex()) maxIndex = itemData.getMaxIndex();
        }
        int num = 0;
        for (int index = 0; index <= maxIndex; ++index) {
            List<List<IndexData>> data = getIndexData(orderColumns, actionResult, index);
            if (CollectionUtils.isNotEmpty(data)) {
                for (List<IndexData> list : data) {
                    for (int i = 0; i < orderColumns.size(); ++i) {
                        String col = orderColumns.get(i);
                        FieldData itemData = actionResult.get(col);
                        if (!itemData.getType().set(table, col, num, list.get(i).getVal())) {
                            log.error("set featureTable fail at algo transform： {} col:{}", name, col);
                        }
                    }
                    num += 1;
                }
            }
        }
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        List<DataResult> result = getDataResultByNames(algoTransform.getFeature(), context);
        result.addAll(getDataResultByNames(algoTransform.getAlgoTransform(), context));
        FeatureTable featureTable = new FeatureTable(name, resFields, ArrowAllocator.getAllocator());
        Map<String, DataResult> dataResultMap = getDataResults(result);
        Map<String, Map<String, DataTypeEnum>> columnMaps = Maps.newHashMap();
        Map<String, Map<String, Field>> fieldMaps = Maps.newHashMap();
        if (CollectionUtils.isNotEmpty(algoTransform.getFeature())) {
            for (String name : algoTransform.getFeature()) {
                columnMaps.put(name, taskFlowConfig.getFeatures().get(name).getColumnMap());
                fieldMaps.put(name, taskFlowConfig.getFeatures().get(name).getFieldMap());
            }
        }
        if (CollectionUtils.isNotEmpty(algoTransform.getAlgoTransform())) {
            for (String name : algoTransform.getAlgoTransform()) {
                columnMaps.put(name, taskFlowConfig.getAlgoTransforms().get(name).getColumnMap());
                fieldMaps.put(name, taskFlowConfig.getAlgoTransforms().get(name).getFieldMap());
            }
        }
        for (FieldAction fieldAction : algoTransform.getActionList()) {
            List<FieldInfo> fields = fieldAction.getFields();
            List<String> input = fieldAction.getInput();
            List<FieldData> fieldDatas = Lists.newArrayList();
            if (CollectionUtils.isNotEmpty(fields)) {
                for (FieldInfo field : fields) {
                    DataResult dataResult = dataResultMap.get(field.getTable());
                    List<Object> itemData = dataResult.get(field.getFieldName());
                    fieldDatas.add(FieldData.of(field.toString(),
                            columnMaps.get(field.getTable()).get(field.getFieldName()),
                            fieldMaps.get(field.getTable()).get(field.getFieldName()),itemData));
                }
            }
            if (CollectionUtils.isNotEmpty(input)) {
                for (String item : input) {
                    FieldData itemData = actionResult.get(item);
                    Assert.notNull(itemData, "use function result, no found result at input: " + item);
                    fieldDatas.add(itemData);
                }
            }
            List<FieldData> res = Lists.newArrayList();
            for (int i = 0; i < fieldAction.getNames().size(); ++i) {
                Field field = getField(fieldAction.getNames().get(i), fieldAction.getTypes().get(i));
                res.add(FieldData.of(fieldAction.getNames().get(i), getType(fieldAction.getTypes().get(i)), field));
                actionResult.put(fieldAction.getNames().get(i), res.get(i));
            }
            if (StringUtils.isEmpty(fieldAction.getFunc())) {
                Assert.isTrue(fieldDatas.size() >= res.size(), "no func need enough input field!");
                for (int i = 0; i < res.size(); ++i) {
                    res.get(i).setIndexValue(fieldDatas.get(i).getIndexValue());
                }
            } else {
                Function function = additionFunctions.get(fieldAction.getFunc());
                if (function == null) {
                    function = taskServiceRegister.getFunction(fieldAction.getFunc());
                    if (function == null) {
                        throw new RuntimeException("function get fail at " + fieldAction.getFunc());
                    }
                }
                if (!function.process(getFieldDataList(fieldDatas), res, fieldAction)) {
                    throw new RuntimeException("the function process fail. func:" + fieldAction.getFunc());
                }
            }
        }
        setTableData(algoTransform.getColumnNames(), actionResult, featureTable);
        featureTable.finish();
        return transform(featureTable, context);
    }

    protected DataResult transform(FeatureTable featureTable, DataContext context) {
        DataResult dataResult = new DataResult();
        dataResult.setFeatureTable(featureTable);
        dataResult.setDataTypes(dataTypes);
        return dataResult;
    }

    public void setFieldData(FeatureTable featureTable, String col, DataTypeEnum dataType, List<Object> data) {
        if (!dataType.set(featureTable, col, data)) {
            log.error("set featureTable fail!");
        }
    }

    public FeatureTable convFeatureTable(String name, List<String> columns, List<FieldData> fields) {
        Map<String, FieldData> fieldDatas = Maps.newHashMap();
        List<FieldData> inputData = Lists.newArrayList();
        for (FieldData fieldItem : fields) {
            fieldDatas.put(fieldItem.getName(), fieldItem);
        }
        for (String col : columns) {
            FieldData item = fieldDatas.get(col);
            Assert.isTrue(item != null, "columns col must in fields, col: " + col);
            inputData.add(item);
        }
        return convFeatureTable(name, inputData);
    }

    public FeatureTable convFeatureTable(String name, List<FieldData> fields) {
        List<Field> inferenceFields = fields.stream().map(x -> new Field(x.getName(), x.getType().getType(), x.getType().getChildFields()))
                .collect(Collectors.toList());
        FeatureTable featureTable = new FeatureTable(name, inferenceFields, ArrowAllocator.getAllocator());
        for (FieldData fieldData : fields) {
            if (!fieldData.getType().set(featureTable, fieldData.getName(), fieldData.getValue())) {
                log.error("set featureTable fail! convFeatureTable at {}", name);
            }
        }
        featureTable.finish();
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
