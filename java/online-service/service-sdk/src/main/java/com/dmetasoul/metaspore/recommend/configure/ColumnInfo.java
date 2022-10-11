package com.dmetasoul.metaspore.recommend.configure;

import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.Types;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.Validate;

import java.util.List;
import java.util.Map;

@Data
@Slf4j
public class ColumnInfo {
    protected List<String> columnNames;
    protected Map<String, DataTypeEnum> columnMap;
    protected Map<String, Field> fieldMap;

    public void setColumns(List<Map<String, Object>> columns) {
        if (CollectionUtils.isNotEmpty(columns)) {
            this.columnNames = Lists.newArrayList();
            this.columnMap = Maps.newHashMap();
            this.fieldMap = Maps.newHashMap();
            columns.forEach(map -> map.forEach((x, y) -> {
                columnNames.add(x);
                DataTypeEnum type = getType(y);
                Validate.notNull(type, "config columns type must be support, type：" + y);
                this.columnMap.put(x, type);
                this.fieldMap.put(x, getField(x, y));
            }));
        }
    }

    @SuppressWarnings("unchecked")
    public static DataTypeEnum getType(Object info) {
        if (info instanceof String) {
            return DataTypes.getDataType((String) info);
        } else if (info instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) info;
            Validate.isTrue(map.size() == 1, "one column one type");
            for (Map.Entry<String, Object> entry : map.entrySet()) {
                String key = entry.getKey();
                return DataTypes.getDataType(key);
            }
        }
        throw new IllegalStateException("column config is wrong");
    }

    @SuppressWarnings("unchecked")
    public static Field getField(String name, Object info) {
        if (info instanceof String) {
            DataTypeEnum typeEnum = DataTypes.getDataType((String) info);
            Validate.notNull(typeEnum, "config columns type must be support, type：" + info);
            Validate.isTrue(!typeEnum.needChildren(), "type need children!");
            return new Field(name, typeEnum.getType(), typeEnum.getChildFields());
        } else if (info instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) info;
            Validate.isTrue(map.size() == 1, "one column one type");
            for (Map.Entry<String, Object> entry : map.entrySet()) {
                String key = entry.getKey();
                DataTypeEnum typeEnum = DataTypes.getDataType(key);
                Validate.notNull(typeEnum, "config columns type must be support, type：" + info);
                Validate.isTrue(typeEnum.needChildren(), "column info type config wrong!");
                Validate.isInstanceOf(Map.class, entry.getValue(), "struct children is map");
                List<Field> children = Lists.newArrayList();
                for (Map.Entry<String, Object> entryItem : ((Map<String, Object>) entry.getValue()).entrySet()) {
                    children.add(getField(entryItem.getKey(), entryItem.getValue()));
                }
                if (CollectionUtils.isEmpty(typeEnum.getChildFields())) {
                    return new Field(name, typeEnum.getType(), children);
                } else {
                    List<Field> sons = Lists.newArrayList();
                    boolean onlyOne = false;
                    for (Field field : typeEnum.getChildFields()) {
                        Types.MinorType minorType = Types.getMinorTypeForArrowType(field.getType());
                        if (minorType == Types.MinorType.STRUCT && CollectionUtils.isEmpty(field.getChildren())) {
                            Validate.isTrue(!onlyOne, "has already empty struct field!");
                            sons.add(new Field(field.getName(), field.getFieldType(), children));
                            onlyOne = true;
                        } else {
                            sons.add(field);
                        }
                    }
                    Validate.isTrue(onlyOne, "must has empty struct field!");
                    return new Field(name, typeEnum.getType(), sons);
                }
            }
        }
        throw new IllegalStateException("column config is wrong");
    }

    public static Field copy(String name, Field field) {
        return new Field(name, field.getFieldType(), field.getChildren());
    }

    public List<String> getColumnNames() {
        return columnNames;
    }
}
