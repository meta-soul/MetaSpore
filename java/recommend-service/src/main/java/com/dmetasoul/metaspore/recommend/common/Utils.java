package com.dmetasoul.metaspore.recommend.common;


import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.google.common.collect.Maps;
import io.milvus.param.R;
import org.springframework.util.StringUtils;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Utils {

    private static double EPSILON = 0.001;

    public static int parseIntFromString(String str, int defaultValue) {
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
