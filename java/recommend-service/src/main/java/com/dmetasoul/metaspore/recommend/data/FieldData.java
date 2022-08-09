package com.dmetasoul.metaspore.recommend.data;

import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import lombok.Data;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;

@Data
public class FieldData {
    private String name;
    private DataTypeEnum type;
    private List<Object> value;

    public static FieldData of(String name, DataTypeEnum type, List<Object> value) {
        FieldData fieldData = new FieldData();
        fieldData.setName(name);
        fieldData.setType(type);
        fieldData.setValue(value);
        return fieldData;
    }

    @SuppressWarnings("unchecked")
    public <T> List<T> getValue() {
        return (List<T>) value;
    }

    public boolean isInvalid() {
        return StringUtils.isEmpty(name) || CollectionUtils.isEmpty(value);
    }

    public boolean isMatch(DataTypeEnum dataType) {
        return !isInvalid() && type.equals(dataType);
    }
}
