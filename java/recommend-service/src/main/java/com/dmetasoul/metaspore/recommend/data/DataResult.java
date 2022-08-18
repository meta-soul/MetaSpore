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
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.util.List;
import java.util.Map;
import java.util.Set;

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

    public List<Object> get(String field) {
        if (featureTable == null || featureTable.getVector(field) == null) throw new IllegalArgumentException("featureTable is null or field not exist");
        FieldVector vector = featureTable.getVector(field);
        List<Object> values = Lists.newArrayList();
        for (int i = 0; i < vector.getValueCount(); ++i) {
            values.add(convValue(vector.getField(), vector.getObject(i)));
        }
        return values;
    }

    public Object get(String field, int index) {
        if (featureTable == null || featureTable.getVector(field) == null) return null;
        FieldVector vector = featureTable.getVector(field);
        if (index < vector.getValueCount() && index >= 0) {
            return convValue(vector.getField(), vector.getObject(index));
        }
        return null;
    }

    public FieldVector getVector(String field) {
        if (featureTable == null || featureTable.getVector(field) == null) throw new IllegalArgumentException("featureTable is null or field not exist");
        return featureTable.getVector(field);
    }

    public boolean isNull() {
        return featureTable == null;
    }

    private void mergeVector(String field, String type, FieldVector fieldVector, boolean doDup) {
        FieldVector vector = getVector(field);
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
                DataTypes.getDataType(type).set(featureTable, field, index, convValue(vector.getField(), fieldVector.getObject(i)));
            }
        }
    }

    public void mergeDataResult(DataResult data, Map<String, String> columnMap, boolean isDup) {
        if (this.isNull() || data.isNull()) return;
        for (Map.Entry<String, String> entry : columnMap.entrySet()) {
            String col = entry.getKey();
            FieldVector fieldVector = data.getVector(col);
            if (fieldVector == null) continue;
            mergeVector(col, entry.getValue(), fieldVector, isDup);
        }
    }

    public void mergeDataResult(List<DataResult> data, Map<String, String> columnMap, boolean isDup) {
        for (DataResult dataResult : data) {
            mergeDataResult(dataResult, columnMap, isDup);
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
