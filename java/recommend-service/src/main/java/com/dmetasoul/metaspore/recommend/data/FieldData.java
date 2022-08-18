package com.dmetasoul.metaspore.recommend.data;

import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.google.common.collect.Lists;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.Assert;

import java.util.List;
import java.util.stream.Collectors;

@Data
public class FieldData {
    private String name;
    private DataTypeEnum type;
    private List<IndexData> indexValue;

    private int maxIndex = -1;

    public static FieldData of(String name, DataTypeEnum type) {
        FieldData fieldData = new FieldData();
        fieldData.setName(name);
        fieldData.setType(type);
        return fieldData;
    }

    public static FieldData of(String name, DataTypeEnum type, List<Object> value) {
        FieldData fieldData = FieldData.of(name, type);
        fieldData.setValue(value);
        return fieldData;
    }

    public static FieldData of(String name, DataTypeEnum type, List<Object> value, List<Integer> index) {
        FieldData fieldData = FieldData.of(name, type);
        fieldData.setValue(value, index);
        return fieldData;
    }

    public static IndexData create(int index, Object obj) {
        return new IndexData(index, obj);
    }

    public void setIndexValue(List<IndexData> value) {
        if (CollectionUtils.isNotEmpty(value)) {
            value.forEach(indexData -> {
                if (maxIndex < indexData.getIndex()) maxIndex = indexData.getIndex();
            });
        }
        this.indexValue = value;
    }

    public void setValue(List<Object> value) {
        if (CollectionUtils.isEmpty(value)) return;
        List<IndexData> data = Lists.newArrayList();
        for (int i = 0; i < value.size(); ++i) {
            data.add(new IndexData(i, value.get(i)));
        }
        maxIndex = value.size() - 1;
        this.indexValue = data;
    }

    public void setValue(List<Object> value, List<Integer> index) {
        if (CollectionUtils.isEmpty(value)) return;
        List<IndexData> data = Lists.newArrayList();
        Assert.isTrue(index != null && value.size() == index.size(), "index and value must has same size");
        for (int i = 0; i < value.size(); ++i) {
            data.add(new IndexData(index.get(i), value.get(i)));
            if (maxIndex < index.get(i)) maxIndex = index.get(i);
        }
        this.indexValue = data;
    }

    @SuppressWarnings("unchecked")
    public <T> List<T> getValue() {
        return (List<T>) indexValue.stream().map(IndexData::getVal).collect(Collectors.toList());
    }

    public List<IndexData> getIndexData(int index, int from) {
        List<IndexData> res = Lists.newArrayList();
        if (CollectionUtils.isEmpty(indexValue)) return res;
        for (int i = from; i < indexValue.size(); ++i) {
            IndexData data
                    = indexValue.get(i);
            if (data.getIndex() == index || data.isAggregate()) {
                res.add(data);
            }
        }
        return res;
    }

    public List<IndexData> getIndexData(int index) {
        return getIndexData(index, 0);
    }

    public void addIndexData(IndexData item) {
        if (item == null) return;
        if (indexValue == null) indexValue = Lists.newArrayList();
        if (maxIndex < item.getIndex()) maxIndex = item.getIndex();
        indexValue.add(item);
    }

    public boolean isInvalid() {
        return StringUtils.isEmpty(name) || CollectionUtils.isEmpty(indexValue);
    }

    public boolean isMatch(DataTypeEnum dataType) {
        return !isInvalid() && type.equals(dataType);
    }
}
