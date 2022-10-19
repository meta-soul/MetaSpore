package com.dmetasoul.metaspore.configure;

import com.google.common.collect.Lists;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor
public class FieldInfo {
    String table;
    String fieldName;

    public FieldInfo(String fieldName) {
        this.fieldName = fieldName;
        this.table = null;
    }

    public static FieldInfo create(String str) {
        String[] array = str.split("\\.");
        if (array.length == 2) {
            return new FieldInfo(array[0], array[1]);
        } else if (array.length == 1) {
            return new FieldInfo(null, array[0]);
        }
        return null;
    }

    public static List<FieldInfo> create(List<String> strs) {
        List<FieldInfo> fields = Lists.newArrayList();
        for (String s : strs) {
            fields.add(create(s));
        }
        return fields;
    }

    public String toString() {
        return fieldName;
    }

    @Override
    public int hashCode() {
        return String.format("%s.%s", table, fieldName).hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null) return false;
        if (obj instanceof FieldInfo) {
            FieldInfo field = (FieldInfo) obj;
            if ((fieldName != null && !fieldName.equals(field.getFieldName())) || (fieldName == null && field.getFieldName() != null)) {
                return false;
            } else
                return (table == null || table.equals(field.getTable())) && (table != null || field.getTable() == null);
        }
        return false;
    }
}
