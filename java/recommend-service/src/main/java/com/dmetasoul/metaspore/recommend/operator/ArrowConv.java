package com.dmetasoul.metaspore.recommend.operator;

import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.arrow.vector.types.Types;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.util.JsonStringHashMap;
import org.apache.arrow.vector.util.Text;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.time.LocalDateTime;
import java.util.AbstractMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ArrowConv {
    public interface Processor {
        Object getValue(Object value);
    }

    public interface StructProcessor {
        Object getValue(List<Field> fields, Object value);
    }
    public static Map<Types.MinorType, Processor> ArrowDataCovers = Maps.newHashMap();

    public static Map<Integer, StructProcessor> StructDataCovers = Maps.newHashMap();
    @SuppressWarnings("unchecked")
    public static Object convValue(Field field, Object value) {
        Types.MinorType minorType = Types.getMinorTypeForArrowType(field.getType());
        if (CollectionUtils.isEmpty(field.getChildren())) {
            if (ArrowDataCovers.containsKey(minorType)) {
                return ArrowDataCovers.get(minorType).getValue(value);
            }
        } else if (minorType == Types.MinorType.LIST) {
            if (value == null) return null;
            Assert.isInstanceOf(List.class, value);
            List<Object> result = Lists.newArrayList();
            for (Object obj : (List<Object>) value) {
                result.add(convValue(field.getChildren().get(0), obj));
            }
            return result;
        } else if (minorType == Types.MinorType.MAP) {
            if (value == null) return null;
            Assert.isInstanceOf(List.class, value);
            Map<Object, Object> result = Maps.newHashMap();
            Field entry = field.getChildren().get(0);
            Field key = entry.getChildren().get(0);
            Field val = entry.getChildren().get(1);
            for (JsonStringHashMap<String, Object> obj : (List<JsonStringHashMap<String, Object>>) value) {
                result.put(convValue(key, obj.get(key.getName())), convValue(val, obj.get(val.getName())));
            }
            return result;
        }
        return value;
    }

    public static Object convValue(DataTypeEnum type, Object value) {
        if (type == null) return value;
        if (StructDataCovers.containsKey(type.getId())) {
            return StructDataCovers.get(type.getId()).getValue(type.getChildFields(), value);
        }
        return value;
    }

    public static List<Object> convValue(DataTypeEnum type, List<Object> value) {
        if (CollectionUtils.isEmpty(value) || type == null) return value;
        return value.stream().map(v -> convValue(type, v)).collect(Collectors.toList());
    }

    static {
        ArrowDataCovers.put(Types.MinorType.VARCHAR, value -> {
            if (value instanceof Text) {
                return value.toString();
            }
            return value;
        });
        ArrowDataCovers.put(Types.MinorType.TIMEMILLI, value -> {
            if (value == null) return null;
            LocalDateTime localDateTime = (LocalDateTime) value;
            return localDateTime.toLocalTime();
        });
    }

    static {
        StructDataCovers.put(DataTypeEnum.LIST_ENTRY_STR_DOUBLE.getId(), (fields, value) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields), "children fields must set");
            Field field = fields.get(0);
            List<Field> children = field.getChildren();
            Assert.isTrue(children.size() == 2, "must has key and value");
            Field keyField = children.get(0);
            Field valField = children.get(1);
            if (value instanceof JsonStringHashMap) {
                @SuppressWarnings("unchecked") JsonStringHashMap<String, Object> map = (JsonStringHashMap<String, Object>) value;
                Object key = convValue(keyField, map.get(keyField.getName()));
                Object val = convValue(valField, map.get(valField.getName()));
                return new AbstractMap.SimpleEntry<>(key, val);
            }
            return value;
        });
    }
}
