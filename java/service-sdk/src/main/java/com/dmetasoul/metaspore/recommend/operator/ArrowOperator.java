package com.dmetasoul.metaspore.recommend.operator;

import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.complex.writer.BaseWriter;
import org.apache.arrow.vector.holders.VarCharHolder;
import org.apache.arrow.vector.types.Types;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.springframework.util.Assert;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.apache.arrow.util.Preconditions.checkArgument;

@Slf4j
@Data
public abstract class ArrowOperator {
    public abstract boolean set(FeatureTable featureTable, int index, String col, Object data);

    public VarCharHolder getVarCharHolder(String str, BufferAllocator allocator) {
        byte[] b = str.getBytes(StandardCharsets.UTF_8);
        VarCharHolder vch = new VarCharHolder();
        vch.start = 0;
        vch.end = b.length;
        vch.buffer = allocator.buffer(b.length);
        vch.buffer.setBytes(0, b);
        return vch;
    }

    public <T> void writeField(BaseWriter.ListWriter writer, T item, Field field,
                               FeatureTable featureTable, BufferAllocator allocator) {
        Assert.notNull(writer, "writer must not null");
        Types.MinorType minorType = Types.getMinorTypeForArrowType(field.getType());
        if (item == null) {
            writer.writeNull();
        } else if (minorType == Types.MinorType.VARCHAR) {
            VarCharHolder vch = getVarCharHolder(String.valueOf(item), allocator);
            writer.varChar().write(vch);
            featureTable.addBuffer(vch.buffer);
        } else if (minorType == Types.MinorType.INT) {
            writer.integer().writeInt(Integer.parseInt(String.valueOf(item)));
        } else if (minorType == Types.MinorType.BIGINT) {
            writer.bigInt().writeBigInt(Long.parseLong(String.valueOf(item)));
        } else if (minorType == Types.MinorType.FLOAT4) {
            writer.float4().writeFloat4(Float.parseFloat(String.valueOf(item)));
        } else if (minorType == Types.MinorType.FLOAT8) {
            writer.float8().writeFloat8(Double.parseDouble(String.valueOf(item)));
        } else if (minorType == Types.MinorType.LIST) {
            writeList(writer.list(), (List<?>)item, field.getChildren(), featureTable, allocator);
        } else if (minorType == Types.MinorType.MAP) {
            writeMap(writer.map(), (Map<?, ?>)item, field.getChildren(), featureTable, allocator);
        } else if (minorType == Types.MinorType.STRUCT){
            writeStruct(writer.struct(), item, field.getChildren(), featureTable, allocator);
        }
    }

    public <T> void writeList(BaseWriter.ListWriter writer, List<T> data, List<Field> fields,
                              FeatureTable featureTable, BufferAllocator allocator) {
        Assert.notNull(writer, "list writer must not null");
        Assert.isTrue(fields != null && fields.size() == 1, "list need one child field");
        writer.startList();
        if (CollectionUtils.isNotEmpty(data)) {
            for (T item : data) {
                writeField(writer, item, fields.get(0), featureTable, allocator);
            }
        }
        writer.endList();
    }

    public <K, V> void writeMap(BaseWriter.MapWriter writer, Map<K, V> data, List<Field> fields,
                                FeatureTable featureTable, BufferAllocator allocator) {
        Assert.notNull(writer, "map writer must not null");
        Assert.isTrue(fields != null && fields.size() == 1, "list need one child field");
        Field structField = fields.get(0);
        Types.MinorType minorType = Types.getMinorTypeForArrowType(structField.getType());
        checkArgument(minorType == Types.MinorType.STRUCT && !structField.isNullable(),"Map data should be a non-nullable struct type");
        checkArgument(structField.getChildren().size() == 2,
                "Map data should be a struct with 2 children. Found: %s", fields);
        writer.startMap();
        if (MapUtils.isNotEmpty(data)) {
            for (Map.Entry<K, V> entry : data.entrySet()) {
                K key = entry.getKey();
                V obj = entry.getValue();
                writer.startEntry();
                Assert.notNull(key, "map key must not null");
                writeField(writer.key(), key, structField.getChildren().get(0), featureTable, allocator);
                writeField(writer.value(), obj, structField.getChildren().get(1), featureTable, allocator);
                writer.endEntry();
            }
        }
        writer.endMap();
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public <T> void writeStruct(BaseWriter.StructWriter writer, T data, List<Field> fields,
                                FeatureTable featureTable, BufferAllocator allocator) {
        Assert.notNull(writer, "list writer must not null");
        writer.start();
        if (data == null) {
            writer.writeNull();
        } else {
            Map<String, Field> fieldMap = Maps.newHashMap();
            for (Field field : fields) {
                fieldMap.put(field.getName(), field);
            }
            if (data instanceof Map) {
                for (Map.Entry<String, Object> entry : ((Map<String, Object>) data).entrySet()) {
                    Field field = fieldMap.get(entry.getKey());
                    if (field == null) {
                        continue;
                    }
                    writeField(writer, entry.getValue(), entry.getKey(), field, featureTable, allocator);
                }
            } else {
                for (java.lang.reflect.Field field : data.getClass().getDeclaredFields()) {
                    field.setAccessible(true);
                    String keyName = field.getName();
                    if (!fieldMap.containsKey(keyName)) {
                        continue;
                    }
                    Object value = field.get(data);
                    writeField(writer, value, keyName, fieldMap.get(keyName), featureTable, allocator);
                }
            }
        }
        writer.end();
    }

    public void writeField(BaseWriter.StructWriter writer, Object data, String name, Field field,
                           FeatureTable featureTable, BufferAllocator allocator) {
        Assert.notNull(writer, "writer must not null");
        Types.MinorType minorType = Types.getMinorTypeForArrowType(field.getType());
        if (minorType == Types.MinorType.VARCHAR) {
            if (data == null) {
                writer.varChar(name).writeNull();
            } else {
                VarCharHolder vch = getVarCharHolder(String.valueOf(data), allocator);
                writer.varChar(name).write(vch);
                featureTable.addBuffer(vch.buffer);
            }
        } else if (minorType == Types.MinorType.INT) {
            if (data == null) {
                writer.integer(name).writeNull();
            } else {
                writer.integer(name).writeInt(Integer.parseInt(String.valueOf(data)));
            }
        } else if (minorType == Types.MinorType.BIGINT) {
            if (data == null) {
                writer.bigInt(name).writeNull();
            } else {
                writer.bigInt(name).writeBigInt(Long.parseLong(String.valueOf(data)));
            }
        } else if (minorType == Types.MinorType.FLOAT4) {
            if (data == null) {
                writer.float4(name).writeNull();
            } else {
                writer.float4(name).writeFloat4(Float.parseFloat(String.valueOf(data)));
            }
        } else if (minorType == Types.MinorType.FLOAT8) {
            if (data == null) {
                writer.float8(name).writeNull();
            } else {
                writer.float8(name).writeFloat8(Double.parseDouble(String.valueOf(data)));
            }
        } else if (minorType == Types.MinorType.LIST) {
            if (data == null) {
                writer.list(name).writeNull();
            } else {
                writeList(writer.list(name), (List<?>) data, field.getChildren(), featureTable, allocator);
            }
        } else if (minorType == Types.MinorType.MAP) {
            if (data == null) {
                writer.map(name).writeNull();
            } else {
                writeMap(writer.map(name), (Map<?, ?>) data, field.getChildren(), featureTable, allocator);
            }
        } else {
            if (data == null) {
                writer.struct(name).writeNull();
            } else {
                writeStruct(writer.struct(name), data, field.getChildren(), featureTable, allocator);
            }
        }
    }
}
