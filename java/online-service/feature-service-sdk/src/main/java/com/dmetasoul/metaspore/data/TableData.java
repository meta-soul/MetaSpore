package com.dmetasoul.metaspore.data;

import com.dmetasoul.metaspore.configure.TableInfo;
import com.dmetasoul.metaspore.configure.FieldInfo;
import com.dmetasoul.metaspore.enums.DataTypeEnum;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.Validate;
import org.springframework.util.Assert;

import java.util.List;
import java.util.Map;

@Slf4j
@Data
public class TableData {
    // schema
    protected List<FieldInfo> names;
    protected Map<FieldInfo, Field> dataSchema;
    protected Map<FieldInfo, DataTypeEnum> dataTypes;
    protected List<Map<FieldInfo, Object>> data;

    private Map<String, FieldInfo> fieldNames;

    public TableData() {
        data = Lists.newArrayList();
        names = Lists.newArrayList();
        dataSchema = Maps.newHashMap();
        dataTypes = Maps.newHashMap();
        fieldNames = Maps.newHashMap();
    }

    public TableData(List<String> names, List<Object> types) {
        data = Lists.newArrayList();
        this.names = Lists.newArrayList();
        dataSchema = Maps.newHashMap();
        dataTypes = Maps.newHashMap();
        fieldNames = Maps.newHashMap();
        addFieldList(names, types);
    }

    public void addFieldList(List<String> names, List<Object> types) {
        if (CollectionUtils.isNotEmpty(names) && names.size() == types.size()) {
            for (int i = 0; i < names.size(); ++i) {
                FieldInfo fieldInfo = getFieldInfo(names.get(i));
                addFieldInfo(fieldInfo);
                addTypeInfo(fieldInfo, types.get(i));
            }
        }
    }

    public void reset(TableData other) {
        data = other.data;
        names = other.names;
        dataSchema = other.dataSchema;
        dataTypes = other.dataTypes;
        fieldNames = other.fieldNames;
    }

    public FieldInfo getFieldInfo(String name) {
        FieldInfo fieldInfo = fieldNames.get(name);
        if (fieldInfo == null) {
            return new FieldInfo(name);
        }
        return fieldInfo;
    }

    public DataTypeEnum getType(String name) {
        return getType(getFieldInfo(name));
    }

    public Field getField(String name) {
        return getField(getFieldInfo(name));
    }

    public DataTypeEnum getType(FieldInfo fieldInfo) {
        if (fieldInfo == null) return null;
        return dataTypes.get(fieldInfo);
    }

    public Field getField(FieldInfo fieldInfo) {
        if (fieldInfo == null) return null;
        return dataSchema.get(fieldInfo);
    }

    public FeatureTable getFeatureTable(String name, List<Field> resFields) {
        FeatureTable featureTable = new FeatureTable(name, resFields);
        for (Field field : resFields) {
            String col = field.getName();
            FieldInfo fieldInfo = getFieldInfo(col);
            Assert.notNull(fieldInfo, "result field name is not exist or duplicate");
            DataTypeEnum dataType = getType(fieldInfo);
            if (dataType == null) {
                log.info("############dataschema: {}", dataSchema);
            }
            Assert.notNull(dataType, "col has type at：" + col);
            for (int index = 0; index < data.size(); ++index) {
                Map<FieldInfo, Object> map = data.get(index);
                if (!dataType.set(featureTable, col, index, map.get(fieldInfo))) {
                    log.error("set featureTable fail at tableData!");
                }
            }
        }
        featureTable.finish();
        return featureTable;
    }

    public DataResult getDataResult(String name, List<Field> resFields) {
        DataResult result = new DataResult();
        if (CollectionUtils.isEmpty(resFields)) {
            return result;
        }
        FeatureTable featureTable = getFeatureTable(name, resFields);
        result.setFeatureTable(featureTable);
        List<DataTypeEnum> dataTypes = Lists.newArrayList();
        for (Field field : resFields) {
            String col = field.getName();
            DataTypeEnum dataType = getType(col);
            Assert.notNull(dataType, "col has type at：" + col);
            dataTypes.add(dataType);
        }
        result.setDataTypes(dataTypes);
        return result;
    }

    public boolean checkFieldInfo(FieldInfo fieldInfo) {
        if (!dataSchema.containsKey(fieldInfo)) {
            return false;
        }
        if (!dataTypes.containsKey(fieldInfo)) {
            return false;
        }
        return true;
    }

    public void addFieldInfo(FieldInfo fieldInfo) {
        if (dataSchema.containsKey(fieldInfo)) {
            return;
        }
        names.add(fieldInfo);
        if (fieldNames.containsKey(fieldInfo.getFieldName())) {
            fieldNames.put(fieldInfo.getFieldName(), null);
        } else {
            fieldNames.put(fieldInfo.getFieldName(), fieldInfo);
        }
    }

    public void addTypeInfo(FieldInfo fieldInfo, Object type) {
        DataTypeEnum dataTypeEnum = TableInfo.getType(type);
        Field field = TableInfo.getField(fieldInfo.getFieldName(), type);
        dataTypes.put(fieldInfo, dataTypeEnum);
        dataSchema.put(fieldInfo, field);
    }

    public void addDataResult(String name, DataResult result) {
        if (result == null || result.isNull()) return;
        List<Field> fields = result.getFields();
        for (int k = 0; k < fields.size(); ++k) {
            Field field = fields.get(k);
            FieldInfo fieldInfo = new FieldInfo(name, field.getName());
            addFieldInfo(fieldInfo);
            dataTypes.put(fieldInfo, result.getDataTypes().get(k));
            dataSchema.put(fieldInfo, field);
            List<Object> value = result.get(k);
            for (int i = 0; i < value.size(); ++i) {
                if (i < data.size()) {
                    data.get(i).put(fieldInfo, value.get(i));
                } else {
                    Map<FieldInfo, Object> item = Maps.newHashMap();
                    item.put(fieldInfo, value.get(i));
                    data.add(item);
                }
            }
        }
    }

    public void addDataResultList(List<String> names, List<DataResult> dataResults) {
        if (CollectionUtils.isNotEmpty(dataResults) && CollectionUtils.isNotEmpty(names)) {
            Validate.isTrue(dataResults.size() == names.size(), "name and dataresults must same size");
            for (int i = 0; i < names.size(); ++i) {
                addDataResult(names.get(i), dataResults.get(i));
            }
        }
    }

    public void copyField(FieldInfo from, String to) {
        if (from == null || StringUtils.isEmpty(to) || !dataSchema.containsKey(from)) return;
        FieldInfo fieldInfo = new FieldInfo(to);
        if (from.equals(fieldInfo) || dataSchema.containsKey(fieldInfo)) return;
        addFieldInfo(fieldInfo);
        dataSchema.put(fieldInfo, dataSchema.get(fieldInfo));
        dataTypes.put(fieldInfo, dataTypes.get(fieldInfo));
        for (Map<FieldInfo, Object> item : data) {
            if (item.containsKey(from)) {
                item.put(fieldInfo, item.get(from));
            }
        }
    }

    public void copyField(String from, String to) {
        FieldInfo fieldInfo = new FieldInfo(from);
        copyField(fieldInfo, to);
    }

    public void addValue(FieldInfo fieldInfo, Object value) {
        addValueList(fieldInfo, List.of(value));
    }

    public void addValueList(String fieldInfo, List<Object> value) {
        addValueList(new FieldInfo(fieldInfo), value);
    }

    public void addValueList(FieldInfo fieldInfo, List<Object> value) {
        Validate.isTrue(checkFieldInfo(fieldInfo), "fieldInfo must add before add value at:" + fieldInfo.getFieldName());
        for (int i = 0; i < value.size(); ++i) {
            if (i < data.size()) {
                data.get(i).put(fieldInfo, value.get(i));
            } else {
                Map<FieldInfo, Object> item = Maps.newHashMap();
                item.put(fieldInfo, value.get(i));
                data.add(item);
            }
        }
    }

    public Object getValue(int index, String name) {
        FieldInfo fieldInfo = getFieldInfo(name);
        return getValue(index, fieldInfo);
    }

    public Object getValue(int index, FieldInfo fieldInfo) {
        return getValue(index, fieldInfo, null);
    }

    public Object getValue(int index, String name, Object default_value) {
        FieldInfo fieldInfo = getFieldInfo(name);
        return getValue(index, fieldInfo, default_value);
    }

    public Object getValue(int index, FieldInfo fieldInfo, Object default_value) {
        if (data.size() <= index || fieldInfo == null || !dataSchema.containsKey(fieldInfo)) {
            return null;
        }
        return data.get(index).getOrDefault(fieldInfo, default_value);
    }

    public void setValue(int index, String name, Object value) {
        FieldInfo fieldInfo = getFieldInfo(name);
        setValue(index, fieldInfo, value);
    }

    public void setValue(int index, FieldInfo fieldInfo, Object value) {
        if (fieldInfo == null) {
            return;
        }
        Validate.isTrue(checkFieldInfo(fieldInfo), "fieldInfo must add before add value at:" + fieldInfo.getFieldName());
        if (data.size() <= index) {
            Map<FieldInfo, Object> item = Maps.newHashMap();
            item.put(fieldInfo, value);
            data.add(item);
        } else {
            data.get(index).put(fieldInfo, value);
        }
    }

    public List<Object> getValueList(FieldInfo input) {
        List<Object> res = Lists.newArrayList();
        for (int i = 0; i < data.size(); ++i) {
            res.add(getValue(i, input));
        }
        return res;
    }
}
