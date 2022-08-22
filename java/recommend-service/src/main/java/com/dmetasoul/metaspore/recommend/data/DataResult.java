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
package com.dmetasoul.metaspore.recommend.data;

import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.recommend.interfaces.MergeOperator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.collections4.SetUtils;
import org.springframework.util.Assert;

import java.util.*;
import java.util.stream.Collectors;

import static com.dmetasoul.metaspore.recommend.operator.ArrowConv.convValue;

/**
 * 用于保存服务结果
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
public class DataResult {
    protected String name;
    protected String reqSign;
    protected FeatureTable featureTable;
    protected List<DataTypeEnum> dataTypes;

    public List<Object> get(String field) {
        if (featureTable == null || featureTable.getVector(field) == null)
            throw new IllegalArgumentException("featureTable is null or field not exist");
        FieldVector vector = featureTable.getVector(field);
        List<Object> values = Lists.newArrayList();
        for (int i = 0; i < vector.getValueCount(); ++i) {
            values.add(convValue(vector.getField(), vector.getObject(i)));
        }
        return values;
    }

    @SuppressWarnings("unchecked")
    public <T> T get(String field, int index) {
        if (featureTable == null || featureTable.getVector(field) == null) return null;
        FieldVector vector = featureTable.getVector(field);
        if (index < vector.getValueCount() && index >= 0) {
            return (T) convValue(vector.getField(), vector.getObject(index));
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    public <T> T get(int field, int index) {
        if (featureTable == null || featureTable.getVector(field) == null) return null;
        FieldVector vector = featureTable.getVector(field);
        if (index < vector.getValueCount() && index >= 0) {
            return (T) convValue(vector.getField(), vector.getObject(index));
        }
        return null;
    }

    public List<Field> getFields() {
        if (featureTable == null || featureTable.getSchema() == null) return List.of();
        return featureTable.getSchema().getFields();
    }

    public FieldVector getVector(String field) {
        if (featureTable == null || featureTable.getVector(field) == null)
            throw new IllegalArgumentException("featureTable is null or field not exist");
        return featureTable.getVector(field);
    }

    public boolean isNull() {
        return featureTable == null;
    }

    private void mergeVector(FieldVector vector, DataTypeEnum type, FieldVector fieldVector, boolean doDup) {
        if (vector == null || fieldVector == null || !vector.getField().equals(fieldVector.getField())) return;
        Set<Object> set = Sets.newHashSet();
        if (doDup) {
            for (int i = 0; i < vector.getValueCount(); ++i) {
                set.add(convValue(vector.getField(), vector.getObject(i)));
            }
        }
        for (int i = 0; i < fieldVector.getValueCount(); ++i) {
            if (!set.contains(fieldVector.getObject(i))) {
                int index = featureTable.getRowCount();

            }
        }
    }

    public void mergeDataResult(DataResult data, List<String> dupFields,
                                Map<String, MergeOperator> mergeOperatorMap,
                                Map<String, List<Object>> mergeFieldData,
                                Map<String, Object> option) {
        if (this.isNull() || data.isNull() || CollectionUtils.isEmpty(data.dataTypes)) return;
        for (int i = 0; i < data.getFeatureTable().getRowCount(); ++i) {
            boolean doDup = false;
            int rowCount = mergeFieldData.computeIfAbsent(featureTable.getVector(0).getName(), key->Lists.newArrayList()).size();
            int index = rowCount;
            for (int j = 0; j < rowCount; ++j) {
                if (CollectionUtils.isNotEmpty(dupFields)) {
                    doDup = true;
                    for (String col : dupFields) {
                        List<Object> colData = mergeFieldData.computeIfAbsent(col, key->Lists.newArrayList());
                        Object val1 = Utils.get(colData, j, null);
                        Object val2 = data.get(col, i);
                        if (!Objects.equals(val1, val2)) {
                            doDup = false;
                            break;
                        }
                    }
                }
                if (doDup) {
                    index = j;
                    break;
                }
            }
            if (!doDup) {
                for (int k = 0; k < dataTypes.size(); ++k) {
                    FieldVector vector = featureTable.getVector(k);
                    mergeFieldData.computeIfAbsent(vector.getName(), key->Lists.newArrayList()).add(data.get(vector.getName(), i));
                }
            }
            if (doDup && MapUtils.isNotEmpty(mergeOperatorMap)) {
                for (Map.Entry<String, MergeOperator> entry : mergeOperatorMap.entrySet()) {
                    MergeOperator operator = entry.getValue();
                    Assert.notNull(operator, "merge operator must not null at col:" + entry.getKey());
                    List<Object> colData = mergeFieldData.get(entry.getKey());
                    Object value = operator.merge(colData.get(index), data.get(entry.getKey(), i), option);
                    colData.set(index, value);
                }
            }
        }
    }

    public void mergeDataResult(List<DataResult> data,
                                List<String> dupFields,
                                Map<String, MergeOperator> mergeOperatorMap,
                                Map<String, Object> option) {
        Map<String, List<Object>> mergeFieldData = Maps.newHashMap();
        for (DataResult dataResult : data) {
            mergeDataResult(dataResult, dupFields, mergeOperatorMap, mergeFieldData, option);
        }
        for (int k = 0; k < dataTypes.size(); ++k) {
            FieldVector vector = featureTable.getVector(k);
            dataTypes.get(k).set(featureTable, vector.getName(), mergeFieldData.get(vector.getName()));
        }
        featureTable.finish();
    }
    public void copyDataResult(DataResult data) {
        if (data == null || Objects.requireNonNull(data).isNull() || isNull()) return;
        copyDataResult(data, 0, data.featureTable.getRowCount());
    }
    public void copyDataResult(DataResult data, int from, int to) {
        if (data == null || Objects.requireNonNull(data).isNull() ||
                isNull() || CollectionUtils.isEmpty(dataTypes)) return;
        for (int i = from; i < to; ++i) {
            for (int k = 0; k < dataTypes.size(); ++k) {
                FieldVector fieldVector = featureTable.getVector(k);
                FieldVector dataVector = data.getFeatureTable().getVector(k);
                Assert.isTrue(fieldVector.getField().equals(dataVector.getField()), "schema must same!");
                dataTypes.get(k).set(featureTable, fieldVector.getName(), i, data.get(k, i));
            }
        }
        featureTable.finish();
    }

    public  void orderAndLimit(DataResult data, List<String> orderBy, int limit) {
        if (data == null || Objects.requireNonNull(data).isNull() ||
                isNull() || CollectionUtils.isEmpty(dataTypes)) return;
        List<Integer> ids = Lists.newArrayList();
        for (int i = 0; i < data.featureTable.getRowCount(); ++i) {
            ids.add(i);
        }
        ids.sort((o1, o2) -> {
            int ret = 0;
            for (String col : orderBy) {
                Object val1 = data.get(col, o1);
                Object val2 = data.get(col, o2);
                if (Objects.equals(val1, val2)) {
                    continue;
                }
                if (val1 == null) return -1;
                if (val2 == null) return 1;
                Assert.isInstanceOf(Comparable.class, val1, "orderBy col must compareable col:" + col);
                @SuppressWarnings("unchecked") Comparable<Object> c = (Comparable<Object>) val2;
                return c.compareTo(val1);
            }
            return ret;
        });
        int index = 0;
        for (int i : ids) {
            if (index >= limit) {
                break;
            }
            for (int k = 0; k < dataTypes.size(); ++k) {
                FieldVector vector = featureTable.getVector(k);
                dataTypes.get(k).set(featureTable, vector.getName(), index, data.get(k, i));
            }
            index += 1;
        }
    }

    public List<Map<String, Object>> output(List<String> columnNames) {
        List<Map<String, Object>> data = Lists.newArrayList();
        if (CollectionUtils.isEmpty(columnNames)) return data;
        if (isNull()) return data;
        for (int i = 0; i < getFeatureTable().getRowCount(); ++i) {
            Map<String, Object> map = Maps.newHashMap();
            for (String col : columnNames) {
                map.put(col, get(col, i));
            }
            data.add(map);
        }
        return data;
    }

    public List<Map<String, Object>> output() {
        List<Map<String, Object>> data = Lists.newArrayList();
        if (isNull()) return data;
        for (int i = 0; i < getFeatureTable().getRowCount(); ++i) {
            Map<String, Object> map = Maps.newHashMap();
            for (Field field : featureTable.getSchema().getFields()) {
                map.put(field.getName(), get(field.getName(), i));
            }
            data.add(map);
        }
        return data;
    }
}