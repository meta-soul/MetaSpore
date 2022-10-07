package com.dmetasoul.metaspore.common;

import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.operator.ArrowConv;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Field;
import org.junit.Assert;
import org.junit.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.*;
import java.util.stream.Collectors;

import static java.time.ZoneOffset.UTC;

@Slf4j
public class DataTypeTest {

    @SneakyThrows
    @Test
    public void TestPrimaryInArrow() {
        List<DataTypeEnum> types = List.of(
                DataTypeEnum.STRING,
                DataTypeEnum.LONG,
                DataTypeEnum.INT,
                DataTypeEnum.DOUBLE,
                DataTypeEnum.BYTE,
                DataTypeEnum.BOOL,
                DataTypeEnum.DATE,
                DataTypeEnum.TIMESTAMP,
                DataTypeEnum.DECIMAL,
                DataTypeEnum.FLOAT,
                DataTypeEnum.SHORT,
                DataTypeEnum.TIME);

        Map<Integer, List<Object>> datas = Maps.newHashMap();
        datas.put(DataTypeEnum.STRING.getId(), List.of("1234", "abcds"));
        datas.put(DataTypeEnum.LONG.getId(), List.of(0L, -1L, Long.MAX_VALUE));
        datas.put(DataTypeEnum.INT.getId(), List.of(1, -1, Integer.MAX_VALUE));
        datas.put(DataTypeEnum.DOUBLE.getId(), List.of(1e-6, -1.00001, 999999999.9999));
        datas.put(DataTypeEnum.BYTE.getId(), List.of(new byte[]{(byte) 0x12, 0x01}, new byte[]{(byte) 0x13, 0x21}, new byte[]{(byte) 0x10, 0x31}));
        datas.put(DataTypeEnum.BOOL.getId(), List.of(false, true, false));
        datas.put(DataTypeEnum.DATE.getId(), List.of(LocalDateTime.of(2022, 8, 12, 17, 28, 5), LocalDateTime.of(2021, 8, 15, 0, 0, 0), LocalDateTime.of(2021, Calendar.SEPTEMBER, 15, 17, 15)));
        datas.put(DataTypeEnum.TIMESTAMP.getId(), List.of(LocalDateTime.of(2021, 8, 15, 17, 15, 20).atZone(UTC).toInstant().toEpochMilli(), 1L, System.currentTimeMillis()));
        datas.put(DataTypeEnum.DECIMAL.getId(), List.of(new BigDecimal(BigInteger.valueOf(43858324658534L), 4), new BigDecimal(new BigInteger("2456214551342342"), 4), new BigDecimal(BigInteger.valueOf(-31467325736457637L), 4))
                .stream().map(x -> {
                    x.setScale(4);
                    return x;
                }).collect(Collectors.toList()));
        datas.put(DataTypeEnum.FLOAT.getId(), List.of(0.123f, -0.446f, 99999.2423432f));
        datas.put(DataTypeEnum.SHORT.getId(), List.of((short) 2, (short) -3, (short) 15));
        datas.put(DataTypeEnum.TIME.getId(), List.of(LocalTime.of(17, 15, 20), LocalTime.of(0, 0, 0), LocalTime.of(23, 59, 59)));

        String field = "field";
        List<Field> inferenceFields = types.stream().map(type -> new Field(field + type.getId(), type.getType(), type.getChildFields())).collect(Collectors.toList());
        try(FeatureTable featureTable = new FeatureTable("table", inferenceFields)) {
            DataResult result = new DataResult();
            result.setFeatureTable(featureTable);
            for (DataTypeEnum type : types) {
                List<Object> list = datas.getOrDefault(type.getId(), List.of());
                Assert.assertTrue(type.set(featureTable, field + type.getId(), list));
            }
            featureTable.finish();
            for (int i = 0; i < featureTable.getRowCount(); ++i) {
                for (DataTypeEnum type : types) {
                    log.info("index:{}, type:{}, data:{}", i, type, result.get(field + type.getId(), i));
                    List<Object> list = datas.getOrDefault(type.getId(), List.of());
                    if (type.equals(DataTypeEnum.BYTE)) {
                        byte[] res = result.get(field + type.getId(), i);
                        byte[] input = (byte[]) CommonUtils.get(list, i, new byte[]{});
                        for (int k = 9; k < res.length; ++k) {
                            Assert.assertEquals(res[k], input[k]);
                        }
                    } else {
                        if (type.equals(DataTypeEnum.DECIMAL)) {
                            BigDecimal data = result.get(field + type.getId(), i);
                            data = data.setScale(4, 6);
                            Assert.assertEquals("type: " + type, CommonUtils.get(list, i, null), data);
                        } else {
                            Assert.assertEquals("type: " + type, CommonUtils.get(list, i, null), result.get(field + type.getId(), i));
                        }
                    }
                }
            }
        }
    }

    @Test
    public void TestIntInArrow() {
        DataTypeEnum type = DataTypeEnum.INT;
        Integer data = 11;
        Integer data1 = 123;

        String field = "field";
        List<Field> inferenceFields = List.of(new Field("field", type.getType(), type.getChildFields()));
        try (FeatureTable featureTable = new FeatureTable("table", inferenceFields)) {
            DataResult result = new DataResult();
            result.setFeatureTable(featureTable);
            Assert.assertTrue(type.set(featureTable, field, 0, data));
            Assert.assertTrue(type.set(featureTable, field, 1, data1));
            featureTable.finish();
            for (int i = 0; i < featureTable.getRowCount(); ++i) {
                log.info("index:{}, data:{}", i, result.get(field, i));
            }
            Assert.assertEquals(data, result.get(field, 0));
            Assert.assertEquals(data1, result.get(field, 1));
        }
    }

    @Test
    public void TestStringInArrow() {
        DataTypeEnum type = DataTypeEnum.STRING;
        String data = "12abCD_/? ？中国~“《》 .QWW,,.。 ";
        String data1 = "12abCD_/? ？中国~“《》 .QWW,,.。 1";

        String field = "field";
        List<Field> inferenceFields = List.of(new Field("field", type.getType(), type.getChildFields()));
        FeatureTable featureTable = new FeatureTable("table", inferenceFields);
        try (DataResult result = new DataResult()) {
            result.setFeatureTable(featureTable);
            Assert.assertTrue(type.set(featureTable, field, 0, data));
            Assert.assertTrue(type.set(featureTable, field, 1, data1));
            featureTable.finish();
            for (int i = 0; i < featureTable.getRowCount(); ++i) {
                log.info("index:{}, data:{}", i, result.get(field, i));
            }
            Assert.assertEquals(data, result.get(field, 0));
            Assert.assertEquals(data1, result.get(field, 1));
        }
    }

    @Test
    public void TestListSTRInArrow() {
        DataTypeEnum type = DataTypeEnum.LIST_STR;
        List<String> keys = List.of("aaa1123", "bbb1234", "ccc1234");

        String field = "field";
        List<Field> inferenceFields = List.of(new Field("field", type.getType(), type.getChildFields()));
        try(FeatureTable featureTable = new FeatureTable("table", inferenceFields)) {
            DataResult result = new DataResult();
            result.setFeatureTable(featureTable);
            Assert.assertTrue(type.set(featureTable, field, 0, keys));
            featureTable.finish();
            for (int i = 0; i < featureTable.getRowCount(); ++i) {
                List<String> data = result.get(field, i);
                log.info("index:{}, data:{}", i, data);
                Assert.assertEquals(keys.size(), data.size());
                for (int k = 0; k < data.size(); ++k) {
                    Assert.assertEquals(keys.get(k), data.get(k));
                }
            }
        }
    }

    @Test
    public void TestListsInArrow() {
        List<DataTypeEnum> types = List.of(
                DataTypeEnum.LIST_STR,
                DataTypeEnum.LIST_LONG,
                DataTypeEnum.LIST_INT,
                DataTypeEnum.LIST_DOUBLE,
                DataTypeEnum.LIST_FLOAT);
        Map<Integer, List<Object>> datas = Maps.newHashMap();
        datas.put(DataTypeEnum.LIST_STR.getId(), List.of("aaa1123", "bbb1234", "ccc1234"));
        datas.put(DataTypeEnum.LIST_LONG.getId(), List.of(12345L, -123L, 0x4573a28ed5L));
        datas.put(DataTypeEnum.LIST_INT.getId(), List.of(123, -234, 4645235));
        datas.put(DataTypeEnum.LIST_DOUBLE.getId(), List.of(123.56, -123.67, 0x4573a28e));
        datas.put(DataTypeEnum.LIST_FLOAT.getId(), List.of(123.56F, -123.67F, 0.00045F));

        String field = "field";
        List<Field> inferenceFields = types.stream().map(type -> new Field(field + type.getId(), type.getType(), type.getChildFields())).collect(Collectors.toList());
        try(FeatureTable featureTable = new FeatureTable("table", inferenceFields)) {
            DataResult result = new DataResult();
            result.setFeatureTable(featureTable);
            for (DataTypeEnum type : types) {
                List<Object> list = datas.getOrDefault(type.getId(), List.of());
                Assert.assertTrue(type.set(featureTable, field + type.getId(), 0, list));
            }
            featureTable.finish();
            for (DataTypeEnum type : types) {
                log.info("index:{}, type:{}, data:{}", 0, type, result.get(field + type.getId(), 0));
                List<Object> list = datas.getOrDefault(type.getId(), List.of());
                List<Object> data = result.get(field + type.getId(), 0);
                Assert.assertEquals(list.size(), data.size());
                for (int k = 0; k < data.size(); ++k) {
                    if (type.equals(DataTypeEnum.LIST_DOUBLE)) {
                        Assert.assertEquals(Double.parseDouble(String.valueOf(list.get(k))), data.get(k));
                    } else {
                        Assert.assertEquals(list.get(k), data.get(k));
                    }
                }
            }
        }
    }

    @Data
    public static class KVEntry {
        private String key;
        private Double value;
    }

    @SuppressWarnings("unchecked")
    @Test
    public void TestListPairInArrow() {
        DataTypeEnum type = DataTypeEnum.LIST_ENTRY_STR_DOUBLE;
        Map<String, Double> scores = Map.of("aaa1123", 12.9, "bbb1234", 129.12, "ccc1234", 120.04);

        String field = "field";
        List<Field> inferenceFields = List.of(new Field("field", type.getType(), type.getChildFields()));
        try(FeatureTable featureTable = new FeatureTable("table", inferenceFields)) {
            DataResult result = new DataResult();
            result.setFeatureTable(featureTable);
            List<Map.Entry<String, Double>> list = Lists.newArrayList();
            list.addAll(scores.entrySet());
            list.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));
            Assert.assertTrue(type.set(featureTable, field, 0, list));
            featureTable.finish();
            for (int i = 0; i < featureTable.getRowCount(); ++i) {
                log.info("index:{}, data:{}", i, result.get(field, i));
            }
            List<Object> res = (List<Object>) ArrowConv.convValue(inferenceFields.get(0), result.get(field, 0));
            assert res != null;
            for (Object item : res) {
                @SuppressWarnings("unchecked") KVEntry entry = CommonUtils.getObjectFromMap((Map<String, Object>) item, KVEntry.class);
                log.info("get entry key: {}, value: {}", entry.getKey(), entry.getValue());
                Assert.assertEquals(entry.getValue(), scores.get(entry.getKey()));
            }
        }
    }

    @Test
    public void TestMapStrDoubleInArrow() {
        DataTypeEnum type = DataTypeEnum.MAP_STR_DOUBLE;
        Map<String, Double> scores = Map.of("aaa1123", 123.9, "bbb1234", 121.12, "ccc1234", 120.04);

        String field = "field";
        List<Field> inferenceFields = List.of(new Field("field", type.getType(), type.getChildFields()));
        try(FeatureTable featureTable = new FeatureTable("table", inferenceFields)) {
            DataResult result = new DataResult();
            result.setFeatureTable(featureTable);
            Assert.assertTrue(type.set(featureTable, field, 0, scores));
            featureTable.finish();
            for (int i = 0; i < featureTable.getRowCount(); ++i) {
                log.info("map index:{}, data:{}", i, result.get(field, i));
            }

            Map<String, Object> data = result.get(field, 0);
            Assert.assertEquals(data.size(), scores.size());
            for (Map.Entry<String, Object> entry : data.entrySet()) {
                log.info("get entry key: {}, value: {}", entry.getKey(), entry.getValue());
                Assert.assertEquals(entry.getValue(), scores.get(entry.getKey()));
            }
        }
    }

    // @Test
    public void TestArrowMemory() {
        DataTypeEnum type = DataTypeEnum.MAP_STR_DOUBLE;
        Map<String, Double> scores = Map.of("aaa1123", 123.9, "bbb1234", 121.12, "ccc1234", 120.04);

        String field = "field";
        List<Field> inferenceFields = List.of(new Field("field", type.getType(), type.getChildFields()));
        log.info("service used memory: {}", CommonUtils.usedMemory());
        double ratio = CommonUtils.useMemoryRatio();
        log.info("service used memory ratio: {}", ratio);
        for (int i = 0; i < 100000; ++i) {
            FeatureTable featureTable = new FeatureTable("table", inferenceFields);
            DataResult result = new DataResult();
            result.setFeatureTable(featureTable);
            Assert.assertTrue(type.set(featureTable, field, 0, scores));
            featureTable.finish();
            double new_ratio = CommonUtils.useMemoryRatio();
            if (new_ratio - ratio > 0.1) {
                log.info("service at: {} used memory ratio from: {} up to {}", i, ratio, new_ratio);
                log.info("service at: {} used memory: {}", i, CommonUtils.usedMemory());
                ratio = new_ratio;
            }
            featureTable.close();
        }
        log.info("service1 used memory ratio: {}", ratio);
    }

    @Test
    public void TestMapsInArrow() {
        List<DataTypeEnum> types = List.of(
                DataTypeEnum.MAP_STR_STR,
                DataTypeEnum.MAP_STR_INT,
                DataTypeEnum.MAP_STR_LONG,
                DataTypeEnum.MAP_STR_FLOAT);
        Map<Integer, Map<String, Object>> datas = Maps.newHashMap();
        datas.put(DataTypeEnum.MAP_STR_STR.getId(), Map.of("key111", "aaa1123", "key112", "bbb1234", "key113", "ccc1234"));
        datas.put(DataTypeEnum.MAP_STR_LONG.getId(), Map.of("longkey111", 12345L, "longkey112", -123L, "longkey113", 0x4573a28ed5L));
        datas.put(DataTypeEnum.MAP_STR_INT.getId(), Map.of("intkey111", 123, "intkey112", -234, "intkey113", 4645235));
        datas.put(DataTypeEnum.MAP_STR_FLOAT.getId(), Map.of("floatkey111", 123.56F, "floatkey112", -123.67F, "floatkey113", 0.00045F));

        String field = "field";
        List<Field> inferenceFields = types.stream().map(type -> new Field(field + type.getId(), type.getType(), type.getChildFields())).collect(Collectors.toList());
        try(FeatureTable featureTable = new FeatureTable("table", inferenceFields)) {
            DataResult result = new DataResult();
            result.setFeatureTable(featureTable);
            for (DataTypeEnum type : types) {
                Map<String, Object> map = datas.getOrDefault(type.getId(), Map.of());
                Assert.assertTrue(type.set(featureTable, field + type.getId(), 0, map));
            }
            featureTable.finish();
            for (DataTypeEnum type : types) {
                log.info("index:{}, type:{}, data:{}", 0, type, result.get(field + type.getId(), 0));
                Map<String, Object> map = datas.getOrDefault(type.getId(), Map.of());
                Map<String, Object> data = result.get(field + type.getId(), 0);
                Assert.assertEquals(map.size(), data.size());
                for (Map.Entry<String, Object> entry : data.entrySet()) {
                    log.info("get entry key: {}, value: {}", entry.getKey(), entry.getValue());
                    Assert.assertEquals(entry.getValue(), map.get(entry.getKey()));
                }
            }
        }
    }
}
