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

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;

import java.math.BigDecimal;
import java.sql.Blob;
import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
/**
 * 用于保存数据字段类型与Class信息的映射关系
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
public class DataTypes {

    @Data
    @AllArgsConstructor
    public static class DataType {
        Class cls;
        ArrowType.PrimitiveType type;
    }
    private static Map<String, DataType> dataTypes = new HashMap<>();
    private static Map<String, Class> complexTypes = new HashMap<>();

    public static boolean typeIsSupport(String type) {
        return dataTypes.containsKey(type) || complexTypes.containsKey(type);
    }
    public static DataType getDataType(String name) {
        return dataTypes.get(name);
    }

    public static Class getDataClass(String name) {
        DataType dataType = dataTypes.get(name);
        if (dataType != null) {
            return dataType.getCls();
        }
        return complexTypes.getOrDefault(name, String.class);
    }

    public static void setDataType(String name, DataType dataType) {
        DataTypes.dataTypes.put(name, dataType);
    }

    static {
        // default
        dataTypes.put("default", new DataType(String.class, ArrowType.Utf8.INSTANCE));

        // Java to Java
        dataTypes.put("string", new DataType(String.class, ArrowType.Utf8.INSTANCE));
        dataTypes.put("long", new DataType(Long.class, new ArrowType.Int(64, true)));
        dataTypes.put("Integer", new DataType(Integer.class, new ArrowType.Int(32, true)));
        dataTypes.put("String", new DataType(String.class, ArrowType.Utf8.INSTANCE));
        dataTypes.put("Long", new DataType(Long.class, new ArrowType.Int(64, true)));
        dataTypes.put("Double", new DataType(Double.class, new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)));

        // Mysql to Java
        dataTypes.put("bigint", new DataType(Long.class, new ArrowType.Int(64, true)));
        dataTypes.put("binary", new DataType(Byte.class, ArrowType.Binary.INSTANCE));
        dataTypes.put("bit", new DataType(Boolean.class, ArrowType.Bool.INSTANCE));
        dataTypes.put("blob", new DataType(Blob.class, ArrowType.LargeBinary.INSTANCE));
        dataTypes.put("char", new DataType(String.class, ArrowType.Utf8.INSTANCE));
        dataTypes.put("date", new DataType(Date.class, new ArrowType.Date(DateUnit.DAY)));
        dataTypes.put("datetime", new DataType(Timestamp.class, new ArrowType.Date(DateUnit.MILLISECOND)));
        dataTypes.put("decimal", new DataType(BigDecimal.class, new ArrowType.Decimal(60, 4, 64)));
        dataTypes.put("double", new DataType(Double.class, new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)));
        dataTypes.put("float", new DataType(Float.class, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)));
        dataTypes.put("int", new DataType(Integer.class, new ArrowType.Int(32, true)));
        dataTypes.put("longblob", new DataType(Blob.class, ArrowType.LargeBinary.INSTANCE));
        dataTypes.put("smallint", new DataType(Short.class, new ArrowType.Int(16, true)));
        dataTypes.put("text", new DataType(String.class, ArrowType.Utf8.INSTANCE));
        dataTypes.put("time", new DataType(Time.class, new ArrowType.Time(TimeUnit.SECOND, 32)));
        dataTypes.put("timestamp", new DataType(Timestamp.class, new ArrowType.Timestamp(TimeUnit.SECOND, "Asia/Shanghai")));
        dataTypes.put("tinyint", new DataType(Byte.class, new ArrowType.Int(8, true)));
        dataTypes.put("varchar", new DataType(String.class, ArrowType.Utf8.INSTANCE));

        dataTypes.put("bool", new DataType(Boolean.class, ArrowType.Bool.INSTANCE));
        dataTypes.put("str", new DataType(String.class, ArrowType.Utf8.INSTANCE));

        complexTypes.put("str[]", List.class);
        complexTypes.put("int[]", List.class);
        complexTypes.put("double[]", List.class);
        complexTypes.put("float[]", List.class);
        complexTypes.put("long[]", List.class);
        complexTypes.put("list_str", List.class);
        complexTypes.put("list_int", List.class);
        complexTypes.put("list_double", List.class);
        complexTypes.put("list_float", List.class);
        complexTypes.put("list_long", List.class);
        complexTypes.put("map_str_str", Map.class);
        complexTypes.put("map_str_int", Map.class);
        complexTypes.put("map_str_double", Map.class);
        complexTypes.put("map_str_float", Map.class);
        complexTypes.put("list", List.class);
        complexTypes.put("map", Map.class);
    }
}