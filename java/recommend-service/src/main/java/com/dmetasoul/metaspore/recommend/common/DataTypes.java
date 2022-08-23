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

import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import lombok.extern.slf4j.Slf4j;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 用于保存数据字段类型与Class信息的映射关系
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
public class DataTypes {
    private static final Map<String, DataTypeEnum> dataTypes = new HashMap<>();
    public static boolean typeIsSupport(String type) {
        return dataTypes.containsKey(type);
    }
    public static DataTypeEnum getDataType(String name) {
        return dataTypes.get(name);
    }

    @SuppressWarnings("rawtypes")
    public static Class getDataClass(String name) {
        DataTypeEnum dataType = dataTypes.get(name);
        return Objects.requireNonNullElse(dataType, DataTypeEnum.STRING).getCls();
    }

    static {
        // default
        dataTypes.put("default", DataTypeEnum.STRING);

        // Java to Java
        dataTypes.put("string", DataTypeEnum.STRING);
        dataTypes.put("long", DataTypeEnum.LONG);
        dataTypes.put("Integer", DataTypeEnum.INT);
        dataTypes.put("String", DataTypeEnum.STRING);
        dataTypes.put("Long", DataTypeEnum.LONG);
        dataTypes.put("Double", DataTypeEnum.DOUBLE);

        // Mysql to Java
        dataTypes.put("bigint", DataTypeEnum.LONG);
        dataTypes.put("binary", DataTypeEnum.BYTE);
        dataTypes.put("bit", DataTypeEnum.BOOL);
        dataTypes.put("blob", DataTypeEnum.BLOB);
        dataTypes.put("char", DataTypeEnum.STRING);
        dataTypes.put("date", DataTypeEnum.DATE);
        dataTypes.put("datetime", DataTypeEnum.TIMESTAMP);
        dataTypes.put("decimal", DataTypeEnum.DECIMAL);
        dataTypes.put("double", DataTypeEnum.DOUBLE);
        dataTypes.put("float", DataTypeEnum.FLOAT);
        dataTypes.put("int", DataTypeEnum.INT);
        dataTypes.put("longblob", DataTypeEnum.BLOB);
        dataTypes.put("smallint", DataTypeEnum.SHORT);
        dataTypes.put("text", DataTypeEnum.STRING);
        dataTypes.put("time", DataTypeEnum.TIME);
        dataTypes.put("timestamp", DataTypeEnum.TIMESTAMP);
        dataTypes.put("tinyint", DataTypeEnum.BYTE);
        dataTypes.put("varchar", DataTypeEnum.STRING);

        dataTypes.put("bool", DataTypeEnum.BOOL);
        dataTypes.put("str", DataTypeEnum.STRING);

        dataTypes.put("str[]", DataTypeEnum.LIST_STR);
        dataTypes.put("int[]", DataTypeEnum.LIST_INT);
        dataTypes.put("double[]", DataTypeEnum.LIST_DOUBLE);
        dataTypes.put("float[]", DataTypeEnum.LIST_FLOAT);
        dataTypes.put("long[]", DataTypeEnum.LIST_LONG);
        dataTypes.put("list_str", DataTypeEnum.LIST_STR);
        dataTypes.put("list_int", DataTypeEnum.LIST_INT);
        dataTypes.put("list_double", DataTypeEnum.LIST_DOUBLE);
        dataTypes.put("list_float", DataTypeEnum.LIST_FLOAT);
        dataTypes.put("list_long", DataTypeEnum.LIST_LONG);
        dataTypes.put("map_str_str", DataTypeEnum.MAP_STR_STR);
        dataTypes.put("map_str_int", DataTypeEnum.MAP_STR_INT);
        dataTypes.put("map_str_long", DataTypeEnum.MAP_STR_LONG);
        dataTypes.put("map_str_double", DataTypeEnum.MAP_STR_DOUBLE);
        dataTypes.put("map_str_float", DataTypeEnum.MAP_STR_FLOAT);
        // value object only support string, int, long, float, double
        dataTypes.put("list_pair_str_double", DataTypeEnum.LIST_ENTRY_STR_DOUBLE);
    }
}