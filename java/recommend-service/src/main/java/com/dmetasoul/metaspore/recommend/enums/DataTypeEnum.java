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
package com.dmetasoul.metaspore.recommend.enums;

import com.dmetasoul.metaspore.recommend.operator.ArrowOperator;
import com.dmetasoul.metaspore.recommend.operator.ListOperator;
import com.dmetasoul.metaspore.recommend.operator.MapOperator;
import com.dmetasoul.metaspore.recommend.operator.StructOperator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.holders.NullableLargeVarBinaryHolder;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.math.BigDecimal;
import java.sql.Blob;
import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.List;
import java.util.Map;

import static com.dmetasoul.metaspore.recommend.common.ConvTools.parseTimestamp;

@SuppressWarnings("unchecked")
@Slf4j
public enum DataTypeEnum {
    STRING(0, String.class, ArrowType.Utf8.INSTANCE, new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof String)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((VarCharVector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setString(index, (String) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    LONG(1, Long.class, new ArrowType.Int(64, true), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Long)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((BigIntVector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setLong(index, (Long) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    INT(2,Integer.class, new ArrowType.Int(32, true), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Integer)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((IntVector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setInt(index, (Integer) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    DOUBLE(3, Double.class, new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Double)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((Float8Vector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setDouble(index, (Double) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    BYTE(4,Byte.class, ArrowType.Binary.INSTANCE, new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Byte)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((UInt1Vector)featureTable.getVector(col)).setNull(index);
            } else {
                ((UInt1Vector)featureTable.getVector(col)).setSafe(index, (Byte)value);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    BOOL(5, Boolean.class, ArrowType.Bool.INSTANCE, new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Boolean)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((BitVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((BitVector)featureTable.getVector(col)).setSafe(index, ((Boolean)value) ? 1 : 0);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    BLOB(6, Blob.class, ArrowType.LargeBinary.INSTANCE, new ArrowOperator() {
        @SneakyThrows
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Blob)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            LargeVarBinaryVector vector = featureTable.getVector(col);
            Assert.notNull(vector, "blob vector is not null at col:" + col);
            if (value == null) {
                vector.setNull(index);
            } else {
                Blob blob = (Blob) value;
                NullableLargeVarBinaryHolder binHolder = new NullableLargeVarBinaryHolder();
                binHolder.start = 0;
                binHolder.end = blob.length();
                binHolder.buffer = vector.getAllocator().buffer(blob.length());
                binHolder.buffer.setBytes(0, blob.getBytes(0L, (int) blob.length()));
                vector.setSafe(index, binHolder);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    DATE(7, Date.class, new ArrowType.Date(DateUnit.DAY), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Date)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((DateDayVector)featureTable.getVector(col)).setNull(index);
            } else {
                //noinspection deprecation
                ((DateDayVector)featureTable.getVector(col)).setSafe(index, ((Date)value).getDay());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    TIMESTAMP(7, Timestamp.class, new ArrowType.Date(DateUnit.MILLISECOND), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            Timestamp data = parseTimestamp(value);
            if (value != null && data == null) {
                log.error("set featureTable fail! value type is not match Timestamp!");
                return false;
            }
            if (value == null) {
                ((DateMilliVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((DateMilliVector)featureTable.getVector(col)).setSafe(index, data.getTime());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    DECIMAL(8, BigDecimal.class, new ArrowType.Decimal(60, 4, 64), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof BigDecimal)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((DecimalVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((DecimalVector)featureTable.getVector(col)).setSafe(index, (BigDecimal)value);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    FLOAT(9, Float.class, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Float)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((Float4Vector)featureTable.getVector(col)).setNull(index);
            } else {
                ((Float4Vector)featureTable.getVector(col)).setSafe(index, (Float)value);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    SHORT(10, Short.class, new ArrowType.Int(16, true), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Short)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((SmallIntVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((SmallIntVector)featureTable.getVector(col)).setSafe(index, (Short)value);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    TIME(11, Time.class, new ArrowType.Time(TimeUnit.SECOND, 32), new ArrowOperator() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Time)) {
                log.error("set featureTable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((TimeSecVector)featureTable.getVector(col)).setNull(index);
            } else {
                //noinspection deprecation
                ((TimeSecVector)featureTable.getVector(col)).setSafe(index, ((Time)value).getSeconds());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    LIST_INT(13, List.class, Integer.class, ArrowType.List.INSTANCE, new ListOperator<Integer>()),
    LIST_LONG(14, List.class, Long.class, ArrowType.List.INSTANCE, new ListOperator<Long>()),
    LIST_STR(15, List.class, String.class, ArrowType.List.INSTANCE, new ListOperator<String>()),
    LIST_FLOAT(16, List.class, Float.class, ArrowType.List.INSTANCE, new ListOperator<Float>()),
    LIST_DOUBLE(17, List.class, Double.class, ArrowType.List.INSTANCE, new ListOperator<Double>()),
    LIST_OBJ(18, List.class, Object.class, ArrowType.List.INSTANCE, new ListOperator<>()),
    MAP_STR_DOUBLE(19, Map.class, new ArrowType.Map(true), new MapOperator<String, Double>()),
    MAP_STR_STR(20, Map.class, new ArrowType.Map(true), new MapOperator<String, String>()),
    MAP_STR_INT(21, Map.class, new ArrowType.Map(true), new MapOperator<String, Integer>()),
    MAP_STR_LONG(22, Map.class, new ArrowType.Map(true), new MapOperator<String, Long>()),
    MAP_STR_FLOAT(23, Map.class, new ArrowType.Map(true), new MapOperator<String, Float>()),
    MAP_STR_OBJ(24, Map.class, new ArrowType.Map(true), new MapOperator<String, Object>()),
    MAP_INT_OBJ(25, Map.class, new ArrowType.Map(true), new MapOperator<Integer, Object>()),
    MAP_LONG_OBJ(26, Map.class, new ArrowType.Map(true), new MapOperator<Long, Object>()),
    STRUCT(99, Object.class, ArrowType.Struct.INSTANCE, new StructOperator());

    private final Integer id;

    private final Class<?> cls;

    private final ArrowType type;

    private final Class<?> subCls;

    private final ArrowOperator op;

    DataTypeEnum(int id, Class<?> cls, ArrowType type, ArrowOperator op){
        this.id = id;
        this.cls = cls;
        this.type = type;
        this.subCls = null;
        this.op = op;
    }

    DataTypeEnum(int id, Class<?> cls, Class<?> subCls, ArrowType type, ArrowOperator op){
        this.id = id;
        this.cls = cls;
        this.type = type;
        this.subCls = subCls;
        this.op = op;
    }

    public int getId() {
        return id;
    }

    public boolean set(FeatureTable featureTable, String col, List<Object> data) {
        op.init(featureTable);
        if (op.getFeatureTable() == null) return false;
        if (CollectionUtils.isNotEmpty(data)) {
            for (int i = 0; i < data.size(); ++i) {
                Object value = data.get(i);
                if (!this.op.set(i, col, value)) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean set(FeatureTable featureTable, String col, int index, Object data) {
        op.init(featureTable);
        if (op.getFeatureTable() == null) return false;
        return this.op.set(index, col, data);
    }
    public Class<?> getCls() {
        return cls;
    }

    public ArrowType getType() {
        return type;
    }

    public static DataTypeEnum getEnumByCls(Class<?> cls) {
        for (DataTypeEnum e : DataTypeEnum.values()) {
            if (e.getCls().equals(cls)) {
                return e;
            }
        }
        return DataTypeEnum.STRUCT;
    }

    public static DataTypeEnum getEnumByType(ArrowType type) {
        for (DataTypeEnum e : DataTypeEnum.values()) {
            if (e.getType().equals(type)) {
                return e;
            }
        }
        return DataTypeEnum.STRUCT;
    }

    public static DataTypeEnum getEnumById(int id) {
        for (DataTypeEnum e : DataTypeEnum.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        return DataTypeEnum.STRUCT;
    }
}
