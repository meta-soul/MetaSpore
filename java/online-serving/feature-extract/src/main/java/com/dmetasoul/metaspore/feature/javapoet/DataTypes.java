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

package com.dmetasoul.metaspore.feature.javapoet;

import java.util.HashMap;
import java.util.Map;

public class DataTypes {
    private static Map<String, String> dataTypes = new HashMap<>();

    public static Map<String, String> getDataTypes() {
        return dataTypes;
    }

    public static void setDataTypes(Map<String, String> dataTypes) {
        DataTypes.dataTypes = dataTypes;
    }

    static {
        // default
        dataTypes.put("default", "java.lang.String");

        // Java to Java
        dataTypes.put("string", "java.lang.String");
        dataTypes.put("long", "java.lang.Long");
        dataTypes.put("Integer", "java.lang.Integer");
        dataTypes.put("String", "java.lang.String");
        dataTypes.put("Long", "java.lang.Long");
        dataTypes.put("Double", "java.sql.Double");

        // Mysql to Java
        dataTypes.put("bigint", "java.lang.Long");
        dataTypes.put("binary", "java.lang.Byte");
        dataTypes.put("bit", "java.lang.Boolean");
        dataTypes.put("blob", "java.sql.Blob");
        dataTypes.put("char", "java.lang.String");
        dataTypes.put("date", "java.sql.Date");
        dataTypes.put("datetime", "java.sql.Timestamp");
        dataTypes.put("decimal", "java.math.BigDecimal");
        dataTypes.put("double", "java.lang.Double");
        dataTypes.put("float", "java.lang.Float");
        dataTypes.put("int", "java.lang.Integer");
        dataTypes.put("longblob", "java.sql.Blob");
        dataTypes.put("smallint", "java.lang.Short");
        dataTypes.put("text", "java.lang.String");
        dataTypes.put("time", "java.sql.Time");
        dataTypes.put("timestamp", "java.sql.Timestamp");
        dataTypes.put("tinyint", "java.lang.Byte");
        dataTypes.put("varchar", "java.lang.String");

        // MongoDB to Java
        dataTypes.put("List", "java.util.List");

        // Cassandra to Java

        // Redis to Java
    }
}