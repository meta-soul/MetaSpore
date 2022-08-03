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


import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.google.common.collect.Maps;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import org.apache.commons.collections4.MapUtils;
import org.springframework.util.StringUtils;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 工具类
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@SuppressWarnings("rawtypes")
public class Utils {
    public static String genResultKey(String name, String taskName) {
        return String.format("%s_%s", name, taskName);
    }
    public static MetricType getMetricType(int index) {
        if (index < 0 || index >= MetricType.values().length) {
            index = 0;
        }
        return MetricType.values()[index];
    }
    public static <T> T nullThenValue(T value, T defaultValue) {
        return value == null ? defaultValue : value;
    }

    @SuppressWarnings("unchecked")
    public static <T> T getField(Map<String, Object> data, String field, T value) {
        if (MapUtils.isNotEmpty(data) && data.containsKey(field)) return (T) data.getOrDefault(field, value);
        return value;
    }

    public Object getField(Map map, String key) {
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

    public static void handleResponseStatus(R<?> r) {
        if (r.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException(r.getMessage());
        }
    }

    public static double getFinalRetrievalScore(Double originalScore, Double maxScore, int algoLevel) {
        double EPSILON = 0.001;
        return originalScore / (maxScore + EPSILON) + algoLevel;
    }

    public static List<List<Float>> getVectorsFromNpsResult(Map<String, ArrowTensor> nspResultMap, String targetKey) {
        ArrowTensor tensor = nspResultMap.get(targetKey);
        ArrowTensor.FloatTensorAccessor accessor = tensor.getFloatData();
        long[] shape = tensor.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Shape length must equal to 2 (batch, vector dim). shape.length: " + shape.length);
        }
        List<List<Float>> vectors = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            List<Float> vector = new ArrayList<>();
            for (int j = 0; j < shape[1]; j++) {
                vector.add(accessor.get(i, j));
            }
            vectors.add(vector);
        }

        return vectors;
    }

    public static List<Float> getScoresFromNpsResult(Map<String, ArrowTensor> nspResultMap, String targetKey, int targetIndex) {
        ArrowTensor tensor = nspResultMap.get(targetKey);
        ArrowTensor.FloatTensorAccessor accessor = tensor.getFloatData();
        long[] shape = tensor.getShape();
        if (targetIndex < 0 || targetIndex >= shape.length) {
            throw new IllegalArgumentException("Target index is out of shape scope. targetIndex: " + targetIndex);
        }
        List<Float> scores = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            scores.add(accessor.get(i, targetIndex));
        }

        return scores;
    }

    public static Map<String, Object> getObjectToMap(Object obj) throws IllegalAccessException {
        Map<String, Object> map = Maps.newHashMap();
        Class<?> cla = obj.getClass();
        Field[] fields = cla.getDeclaredFields();
        for (Field field : fields) {
            field.setAccessible(true);
            String keyName = field.getName();
            Object value = field.get(obj);
            if (value == null)
                value = "";
            map.put(keyName, value);
        }
        return map;
    }
}
