package com.dmetasoul.metaspore.recommend.operator;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.arrow.vector.types.Types;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.util.JsonStringHashMap;
import org.apache.arrow.vector.util.Text;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.Validate;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
            Validate.isInstanceOf(List.class, value);
            List<Object> result = Lists.newArrayList();
            for (Object obj : (List<Object>) value) {
                result.add(convValue(field.getChildren().get(0), obj));
            }
            return result;
        } else if (minorType == Types.MinorType.MAP) {
            if (value == null) return null;
            Validate.isInstanceOf(List.class, value);
            Map<Object, Object> result = Maps.newHashMap();
            Field entry = field.getChildren().get(0);
            Field key = entry.getChildren().get(0);
            Field val = entry.getChildren().get(1);
            for (JsonStringHashMap<String, Object> obj : (List<JsonStringHashMap<String, Object>>) value) {
                if (obj != null && obj.size() == 2) {
                    result.put(convValue(key, obj.get(key.getName())), convValue(val, obj.get(val.getName())));
                }
            }
            return result;
        } else if (minorType == Types.MinorType.STRUCT) {
            if (value == null) return null;
            Map<String, Object> result = Maps.newHashMap();
            Map<String, Object> obj = (HashMap<String, Object>) value;
            for (Field f : field.getChildren()) {
                result.put(f.getName(), convValue(f, obj.get(f.getName())));
            }
            return result;
        }
        return value;
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
}
