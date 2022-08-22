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

import com.dmetasoul.metaspore.recommend.operator.*;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.holders.*;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import org.apache.arrow.vector.types.pojo.Field;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.sql.Blob;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.List;
import java.util.Map;

import static com.dmetasoul.metaspore.recommend.common.ConvTools.*;
import static java.time.ZoneOffset.UTC;

@SuppressWarnings("unchecked")
@Slf4j
public enum DataTypeEnum {
    STRING(0, String.class, FieldType.nullable(ArrowType.Utf8.INSTANCE), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof String)) {
                log.error("set featureTable fail! value type is not match String value:{}!", value);
                return false;
            }
            VarCharVector vector = featureTable.getVector(col);
            if (value == null) {
                vector.setNull(index);
            } else {
                String data = (String) value;
                byte[] b = data.getBytes(StandardCharsets.UTF_8);
                VarCharHolder vch = new VarCharHolder();
                vch.start = 0;
                vch.end = b.length;
                vch.buffer = vector.getAllocator().buffer(b.length);
                vch.buffer.setBytes(0, b);
                vector.setSafe(index, vch);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    LONG(1, Long.class, FieldType.nullable(new ArrowType.Int(64, true)), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof Long)) {
                log.error("set featureTable fail! value type is not match Long value：{}!", value);
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
    INT(2,Integer.class, FieldType.nullable(new ArrowType.Int(32, true)), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof Integer)) {
                log.error("set featureTable fail! value type is not match Integer value:{}!", value);
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
    DOUBLE(3, Double.class, FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof Double)) {
                log.error("set featureTable fail! value type is not match Double value: {}! ", value);
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
    BYTE(4,Byte.class, FieldType.nullable(ArrowType.Binary.INSTANCE), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof byte[])) {
                log.error("set featureTable fail! value type is not match Byte value:{}!", value);
                return false;
            }
            VarBinaryVector vector = featureTable.getVector(col);
            if (value == null) {
                vector.setNull(index);
            } else {
                VarBinaryHolder binHolder = new VarBinaryHolder();
                binHolder.start = 0;
                byte[] data = (byte[]) value;
                binHolder.end = data.length;
                binHolder.buffer = vector.getAllocator().buffer(data.length);
                binHolder.buffer.setBytes(0, data);
                vector.setSafe(index, binHolder);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    BOOL(5, Boolean.class, FieldType.nullable(ArrowType.Bool.INSTANCE), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof Boolean)) {
                log.error("set featureTable fail! value type is not match Boolean value: {}!", value);
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
    BLOB(6, Blob.class, FieldType.nullable(ArrowType.LargeBinary.INSTANCE), new ArrowOperator() {
        @SneakyThrows
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof Blob)) {
                log.error("set featureTable fail! value type is not match Blob value: {}!", value);
                return false;
            }
            LargeVarBinaryVector vector = featureTable.getVector(col);
            Assert.notNull(vector, "blob vector is not null at col:" + col);
            if (value == null) {
                vector.setNull(index);
            } else {
                Blob blob = (Blob) value;
                LargeVarBinaryHolder binHolder = new LargeVarBinaryHolder();
                binHolder.start = 0;
                binHolder.end = blob.length();
                binHolder.buffer = vector.getAllocator().buffer(blob.length());
                binHolder.buffer.setBytes(0, blob.getBytes(1L, (int) blob.length()));
                vector.setSafe(index, binHolder);
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    DATE(7, LocalDateTime.class, FieldType.nullable(new ArrowType.Date(DateUnit.MILLISECOND)), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            LocalDateTime data = parseLocalDateTime(value);
            if (value != null && data == null) {
                log.error("set featureTable fail! value type is not match Date value: {}, cls: {}, index: {}!", value, value.getClass(), index);
                return false;
            }
            if (value == null) {
                ((DateMilliVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((DateMilliVector)featureTable.getVector(col)).setSafe(index, data.atZone(UTC).toInstant().toEpochMilli());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    TIMESTAMP(8, Timestamp.class, FieldType.nullable(new ArrowType.Timestamp(TimeUnit.SECOND, "Asia/Shanghai")), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            Timestamp data = parseTimestamp(value);
            if (value != null && data == null) {
                log.error("set featureTable fail! value type is not match Timestamp value: {}!", value);
                return false;
            }
            if (value == null) {
                ((TimeStampSecTZVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((TimeStampSecTZVector)featureTable.getVector(col)).setSafe(index, data.getTime());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    DECIMAL(9, BigDecimal.class, FieldType.nullable(new ArrowType.Decimal(60, 4, 64)), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof BigDecimal)) {
                log.error("set featureTable fail! value type is not match BigDecimal value: {}!", value);
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
    FLOAT(10, Float.class, FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof Float)) {
                log.error("set featureTable fail! value type is not match Float value: {}!", value);
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
    SHORT(11, Short.class, FieldType.nullable(new ArrowType.Int(16, true)), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            if (value != null && !(value instanceof Short)) {
                log.error("set featureTable fail! value type is not match Short value: {}!", value);
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
    TIME(12, Time.class, FieldType.nullable(new ArrowType.Time(TimeUnit.MILLISECOND, 32)), new ArrowOperator() {
        @Override
        public boolean set(FeatureTable featureTable, int index, String col, Object value) {
            LocalTime data = parseLocalTime(value);
            if (value != null && data == null) {
                log.error("set featureTable fail! value type is not match Time value : {}!", value);
                return false;
            }
            if (value == null) {
                ((TimeMilliVector)featureTable.getVector(col)).setNull(index);
            } else {
                LocalDateTime localDateTime = LocalDateTime.of(LocalDate.EPOCH, data);
                ((TimeMilliVector)featureTable.getVector(col)).setSafe(index, (int) localDateTime.toInstant(UTC).toEpochMilli());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    LIST_INT(13, List.class, FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.notNullable("item",new ArrowType.Int(32, true))), new ListOperator<Integer>()),
    LIST_LONG(14, List.class, FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.notNullable("item",new ArrowType.Int(64, true))), new ListOperator<Long>()),
    LIST_STR(15, List.class, FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.notNullable("item",ArrowType.Utf8.INSTANCE)), new ListOperator<String>()),
    LIST_FLOAT(16, List.class, FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.notNullable("item",new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE))), new ListOperator<Float>()),
    LIST_DOUBLE(17, List.class, FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.notNullable("item",new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))), new ListOperator<Double>()),
    LIST_ENTRY_STR_DOUBLE(18, List.class, FieldType.nullable(ArrowType.List.INSTANCE), List.of(new Field("item", FieldType.notNullable(ArrowType.Struct.INSTANCE), List.of(
            Field.notNullable("key",ArrowType.Utf8.INSTANCE),
            Field.notNullable("value",new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
    ))), new ListOperator<>()),
    MAP_STR_DOUBLE(19, Map.class, FieldType.nullable(new ArrowType.Map(true)), List.of(
            new Field("entry", FieldType.notNullable(ArrowType.Struct.INSTANCE), List.of(
            Field.notNullable("key",ArrowType.Utf8.INSTANCE),
            Field.notNullable("value",new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
    ))), new MapOperator<String, Double>()),
    MAP_STR_STR(20, Map.class, FieldType.nullable(new ArrowType.Map(true)), List.of(
            new Field("entry", FieldType.notNullable(ArrowType.Struct.INSTANCE), List.of(
                Field.notNullable("key",ArrowType.Utf8.INSTANCE),
                Field.notNullable("value",ArrowType.Utf8.INSTANCE)
    ))), new MapOperator<String, String>()),
    MAP_STR_INT(21, Map.class, FieldType.nullable(new ArrowType.Map(true)), List.of(
            new Field("entry", FieldType.notNullable(ArrowType.Struct.INSTANCE), List.of(
            Field.notNullable("key",ArrowType.Utf8.INSTANCE),
            Field.notNullable("value",new ArrowType.Int(32, true))
    ))), new MapOperator<String, Integer>()),
    MAP_STR_LONG(22, Map.class, FieldType.nullable(new ArrowType.Map(true)), List.of(
            new Field("entry", FieldType.notNullable(ArrowType.Struct.INSTANCE), List.of(
            Field.notNullable("key",ArrowType.Utf8.INSTANCE),
            Field.notNullable("value",new ArrowType.Int(64, true))
    ))), new MapOperator<String, Long>()),
    MAP_STR_FLOAT(23, Map.class, FieldType.nullable(new ArrowType.Map(true)), List.of(
            new Field("entry", FieldType.notNullable(ArrowType.Struct.INSTANCE), List.of(
            Field.notNullable("key",ArrowType.Utf8.INSTANCE),
            Field.notNullable("value",new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE))
    ))), new MapOperator<String, Float>());

    private final Integer id;

    private final Class<?> cls;

    private final FieldType type;

    private final List<Field> childFields;

    private final ArrowOperator op;

    DataTypeEnum(int id, Class<?> cls, FieldType type, ArrowOperator op){
        this.id = id;
        this.cls = cls;
        this.type = type;
        this.childFields = Lists.newArrayList();
        this.op = op;
    }

    DataTypeEnum(int id, Class<?> cls, FieldType type, List<Field> childFields, ArrowOperator op){
        this.id = id;
        this.cls = cls;
        this.type = type;
        this.childFields = childFields;
        this.op = op;
    }

    public int getId() {
        return id;
    }

    public boolean set(FeatureTable featureTable, String col, List<Object> data) {
        if (CollectionUtils.isNotEmpty(data)) {
            for (int i = 0; i < data.size(); ++i) {
                Object value = data.get(i);
                if (!this.op.set(featureTable, i, col, value)) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean set(FeatureTable featureTable, String col, int index, Object data) {
        return this.op.set(featureTable, index, col, data);
    }
    public Class<?> getCls() {
        return cls;
    }

    public FieldType getType() {
        return type;
    }

    public List<Field> getChildFields() {
        return childFields;
    }

    public static DataTypeEnum getEnumByCls(Class<?> cls) {
        for (DataTypeEnum e : DataTypeEnum.values()) {
            if (e.getCls().equals(cls)) {
                return e;
            }
        }
        return DataTypeEnum.STRING;
    }

    public static DataTypeEnum getEnumByType(ArrowType type) {
        for (DataTypeEnum e : DataTypeEnum.values()) {
            if (e.getType().equals(type)) {
                return e;
            }
        }
        return DataTypeEnum.STRING;
    }

    public static DataTypeEnum getEnumById(int id) {
        for (DataTypeEnum e : DataTypeEnum.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        return DataTypeEnum.STRING;
    }
}
