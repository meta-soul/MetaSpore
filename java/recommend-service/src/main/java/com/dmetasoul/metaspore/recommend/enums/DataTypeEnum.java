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

import com.dmetasoul.metaspore.serving.FeatureTable;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.commons.collections4.CollectionUtils;

import java.math.BigDecimal;
import java.sql.Blob;
import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.Collection;
import java.util.List;
import java.util.Map;

@Slf4j
public enum DataTypeEnum {
    STRING(0, String.class, ArrowType.Utf8.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof String)) {
                log.error("set featuraYable fail! value type is not match!");
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
    LONG(1, Long.class, new ArrowType.Int(64, true), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Long)) {
                log.error("set featuraYable fail! value type is not match!");
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
    INT(2,Integer.class, new ArrowType.Int(32, true), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Integer)) {
                log.error("set featuraYable fail! value type is not match!");
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
    DOUBLE(3, Double.class, new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Double)) {
                log.error("set featuraYable fail! value type is not match!");
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
    BYTE(4,Byte.class, ArrowType.Binary.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Byte)) {
                log.error("set featuraYable fail! value type is not match!");
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
    BOOL(5, Boolean.class, ArrowType.Bool.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Boolean)) {
                log.error("set featuraYable fail! value type is not match!");
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
    BLOB(6, Blob.class, ArrowType.LargeBinary.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            log.error("set featuraYable fail! blob type is not support!");
            // to do
            return false;
        }
    }),
    DATE(7, Date.class, new ArrowType.Date(DateUnit.DAY), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Date)) {
                log.error("set featuraYable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((DateDayVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((DateDayVector)featureTable.getVector(col)).setSafe(index, ((Date)value).getDay());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    TIMESTAMP(7, Timestamp.class, new ArrowType.Date(DateUnit.MILLISECOND), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Timestamp)) {
                log.error("set featuraYable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((DateMilliVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((DateMilliVector)featureTable.getVector(col)).setSafe(index, ((Timestamp)value).getTime());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    DECIMAL(8, BigDecimal.class, new ArrowType.Decimal(60, 4, 64), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof BigDecimal)) {
                log.error("set featuraYable fail! value type is not match!");
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
    FLOAT(9, Float.class, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Float)) {
                log.error("set featuraYable fail! value type is not match!");
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
    SHORT(10, Short.class, new ArrowType.Int(16, true), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Short)) {
                log.error("set featuraYable fail! value type is not match!");
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
    TIME(11, Time.class, new ArrowType.Time(TimeUnit.SECOND, 32), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof Time)) {
                log.error("set featuraYable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((TimeSecVector)featureTable.getVector(col)).setNull(index);
            } else {
                ((TimeSecVector)featureTable.getVector(col)).setSafe(index, ((Time)value).getSeconds());
            }
            featureTable.setRowCount(index+1);
            return true;
        }
    }),
    LIST(12, List.class, ArrowType.List.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            log.error("set featuraYable fail! list<?> type is not support!");
            return false;
        }
    }),
    LIST_INT(13, List.class, Integer.class, ArrowType.List.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof List)) {
                log.error("set featuraYable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((ListVector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setIntList(index, (List<Integer>) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    LIST_LONG(14, List.class, Long.class, ArrowType.List.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof List)) {
                log.error("set featuraYable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((ListVector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setLongList(index, (List<Long>) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    LIST_STR(15, List.class, String.class, ArrowType.List.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof List)) {
                log.error("set featuraYable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((ListVector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setStringList(index, (List<String>) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    LIST_FLOAT(16, List.class, Float.class, ArrowType.List.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof List)) {
                log.error("set featuraYable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((ListVector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setFloatList(index, (List<Float>) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    LIST_DOUBLE(17, List.class, Double.class, ArrowType.List.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof List)) {
                log.error("set featuraYable fail! value type is not match!");
                return false;
            }
            if (value == null) {
                ((ListVector)featureTable.getVector(col)).setNull(index);
                featureTable.setRowCount(index+1);
            } else {
                featureTable.setDoubleList(index, (List<Double>) value, featureTable.getVector(col));
            }
            return true;
        }
    }),
    MAP(18, Map.class, new ArrowType.Map(false), new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            log.error("set featuraYable fail! map type is not support!");
            return false;
        }
    }),
    DEFAULT(19, String.class, ArrowType.Utf8.INSTANCE, new Op() {
        @Override
        public boolean set(int index, String col, Object value) {
            if (value != null && !(value instanceof String)) {
                log.error("set featuraYable fail! value type is not match!");
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
    });

    private Integer id;

    private Class<?> cls;

    private ArrowType type;

    private Class<?> subCls;

    private Op op;

    DataTypeEnum(int id, Class<?> cls, ArrowType type, Op op){
        this.id = id;
        this.cls = cls;
        this.type = type;
        this.subCls = null;
        this.op = op;
    }

    DataTypeEnum(int id, Class<?> cls, Class<?> subCls, ArrowType type, Op op){
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
        if (op.featureTable == null) return false;
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

    public boolean set(FeatureTable featureTable, String col, Object data) {
        op.init(featureTable);
        if (op.featureTable == null) return false;
        int index = op.featureTable.g
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
        return DataTypeEnum.DEFAULT;
    }

    public static DataTypeEnum getEnumById(int id) {
        for (DataTypeEnum e : DataTypeEnum.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        return DataTypeEnum.DEFAULT;
    }

    public static abstract class Op {
        protected FeatureTable featureTable;
        public void init(FeatureTable featureTable) {
            this.featureTable = featureTable;
        }
        public abstract boolean set(int index, String col, Object data);
    }
}
