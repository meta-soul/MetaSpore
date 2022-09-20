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

package com.dmetasoul.metaspore.recommend.common;


import com.google.common.collect.Maps;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 工具类
 * Created by @author qinyy907 in 14:24 22/07/15.
 */

@Slf4j
public class CommonUtils {
    public static String genResultKey(String name, String taskName) {
        return String.format("%s_%s", name, taskName);
    }
    public static <T> T nullThenValue(T value, T defaultValue) {
        return value == null ? defaultValue : value;
    }

    @SuppressWarnings("unchecked")
    public static <T> T getField(Map<String, Object> data, String field, T value) {
        if (MapUtils.isNotEmpty(data) && data.containsKey(field)) return (T) data.getOrDefault(field, value);
        return value;
    }

    @SuppressWarnings("unchecked")
    public static <T> T getField(Map<String, Object> data, String field, T value, Class<?> cls) {
        if (MapUtils.isNotEmpty(data) && data.containsKey(field)) {
            Object obj = data.getOrDefault(field, value);
            if (cls.isInstance(obj)) return (T)obj;
            return ConvTools.parseObject(obj, cls);
        }
        return value;
    }

    public static <T> T getField(Map<String, Object> data, String field) {
        return getField(data, field, null);
    }

    public static Object getObject(Map map, String key) {
        if (MapUtils.isNotEmpty(map) && map.containsKey(key)) {
            return map.get(key);
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    public static boolean setFieldFail(Map map, List<String> columnName, int index, Object value) {
        if (index < 0 || columnName == null || map == null || index >= columnName.size()) {
            return true;
        }
        map.put(columnName.get(index), value);
        return false;
    }

    public static int parseIntFromString(String str, int defaultValue) {
        //noinspection deprecation
        if (StringUtils.isEmpty(str)) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(str);
        } catch (Exception e) {
            return defaultValue;
        }
    }
    @SneakyThrows
    public static Map<String, Object> getObjectToMap(Object obj) {
        Map<String, Object> map = Maps.newHashMap();
        Class<?> cla = obj.getClass();
        Field[] fields = cla.getDeclaredFields();
        for (Field field : fields) {
            if (field.trySetAccessible()) {
                String keyName = field.getName();
                Object value = field.get(obj);
                map.put(keyName, value);
            }
        }
        return map;
    }
    @SneakyThrows
    @SuppressWarnings("unchecked")
    public static <T> T getObjectFromMap(Map<String, Object> data, Class<?> cls) {
        Object obj = cls.getConstructor().newInstance();
        Field[] fields = cls.getDeclaredFields();
        for (Field field : fields) {
            if (field.trySetAccessible() && data.containsKey(field.getName())) {
                String keyName = field.getName();
                field.set(obj, data.get(keyName));
            }
        }
        return (T) obj;
    }
    public static <T> T get(List<T> list, int index, T value) {
        if (CollectionUtils.isNotEmpty(list) && index >= 0 && index < list.size()) {
            return list.get(index);
        }
        return value;
    }
    @SuppressWarnings("unchecked")
    public static <T> T getObject(List<Object> list, int index, T value) {
        if (CollectionUtils.isNotEmpty(list) && index >= 0 && index < list.size()) {
            return (T) list.get(index);
        }
        return value;
    }
}
