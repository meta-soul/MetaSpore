package com.dmetasoul.metaspore.recommend.operator;

import com.dmetasoul.metaspore.serving.FeatureTable;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.complex.writer.BaseWriter;
import org.apache.arrow.vector.holders.VarCharHolder;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.springframework.util.Assert;

import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

@Slf4j
@Data
public abstract class ArrowOperator {
    protected FeatureTable featureTable;
    public void init(FeatureTable featureTable) {
        this.featureTable = featureTable;
    }
    public abstract boolean set(int index, String col, Object data);

    public Object getValue(Object value) {
        return value;
    }
    @SuppressWarnings("unchecked")
    public <T> T get(String col, int index) {
        if (featureTable == null || featureTable.getVector(col) == null) return null;
        FieldVector vector = featureTable.getVector(col);
        if (index < vector.getValueCount() && index >= 0) {
            return (T) getValue(vector.getObject(index));
        }
        return null;
    }

    public VarCharHolder getVarCharHolder(String str, BufferAllocator allocator) {
        byte[] b = str.getBytes(StandardCharsets.UTF_8);
        VarCharHolder vch = new VarCharHolder();
        vch.start = 0;
        vch.end = b.length;
        vch.buffer = allocator.buffer(b.length);
        vch.buffer.setBytes(0, b);
        return vch;
    }

    public <T> void writeField(BaseWriter.ListWriter writer, T item, BufferAllocator allocator) {
        Assert.notNull(writer, "writer must not null");
        if (item == null) {
            writer.writeNull();
        } else if (item instanceof String) {
            VarCharHolder vch = getVarCharHolder((String) item, allocator);
            writer.varChar().write(vch);
        } else if (item instanceof Integer) {
            writer.integer().writeInt((Integer) item);
        } else if (item instanceof Long) {
            writer.bigInt().writeBigInt((Long) item);
        } else if (item instanceof Float) {
            writer.float4().writeFloat4((Float) item);
        } else if (item instanceof Double) {
            writer.float8().writeFloat8((Double) item);
        } else if (item instanceof List) {
            writeList(writer.list(), (List<?>)item, allocator);
        } else if (item instanceof Map) {
            writeMap(writer.map(), (Map<?, ?>)item, allocator);
        } else {
            writeStruct(writer.struct(), item, allocator);
        }
    }

    public <T> void writeList(BaseWriter.ListWriter writer, List<T> data, BufferAllocator allocator) {
        Assert.notNull(writer, "list writer must not null");
        writer.startList();
        if (CollectionUtils.isNotEmpty(data)) {
            for (T item : data) {
                writeField(writer, item, allocator);
            }
        }
        writer.endList();
    }

    public <K, V> void writeMap(BaseWriter.MapWriter writer, Map<K, V> data, BufferAllocator allocator) {
        Assert.notNull(writer, "map writer must not null");
        writer.startMap();
        if (MapUtils.isNotEmpty(data)) {
            for (Map.Entry<K, V> entry : data.entrySet()) {
                K key = entry.getKey();
                V obj = entry.getValue();
                writer.startEntry();
                Assert.notNull(key, "map key must not null");
                writeField(writer.key(), key, allocator);
                writeField(writer.value(), obj, allocator);
                writer.endEntry();
            }
        }
        writer.endMap();
    }

    @SneakyThrows
    public <T> void writeStruct(BaseWriter.StructWriter writer, T data, BufferAllocator allocator) {
        Assert.notNull(writer, "list writer must not null");
        writer.start();
        if (data == null) {
            writer.writeNull();
        } else {
            Class<?> cla = data.getClass();
            Field[] fields = cla.getDeclaredFields();
            for (Field field : fields) {
                field.setAccessible(true);
                String keyName = field.getName();
                Object value = field.get(data);
                Class<?> fieldCls = field.getClass();
                writeField(writer, value, keyName, allocator, fieldCls);
            }
        }
        writer.end();
    }

    public void writeField(BaseWriter.StructWriter writer, Object data, String name, BufferAllocator allocator, Class<?> clazz) {
        Assert.notNull(writer, "writer must not null");
        if (String.class.isAssignableFrom(clazz)) {
            if (data == null) {
                writer.varChar(name).writeNull();
            } else {
                VarCharHolder vch = getVarCharHolder((String) data, allocator);
                writer.varChar(name).write(vch);
            }
        } else if (Integer.class.isAssignableFrom(clazz)) {
            if (data == null) {
                writer.integer(name).writeNull();
            } else {
                writer.integer(name).writeInt((Integer) data);
            }
        } else if (Long.class.isAssignableFrom(clazz)) {
            if (data == null) {
                writer.bigInt(name).writeNull();
            } else {
                writer.bigInt(name).writeBigInt((Long) data);
            }
        } else if (Float.class.isAssignableFrom(clazz)) {
            if (data == null) {
                writer.float4(name).writeNull();
            } else {
                writer.float4(name).writeFloat4((Float) data);
            }
        } else if (Double.class.isAssignableFrom(clazz)) {
            if (data == null) {
                writer.float8(name).writeNull();
            } else {
                writer.float8(name).writeFloat8((Double) data);
            }
        } else if (List.class.isAssignableFrom(clazz)) {
            if (data == null) {
                writer.list(name).writeNull();
            } else {
                writeList(writer.list(name), (List<?>) data, allocator);
            }
        } else if (Map.class.isAssignableFrom(clazz)) {
            if (data == null) {
                writer.map(name).writeNull();
            } else {
                writeMap(writer.map(name), (Map<?, ?>) data, allocator);
            }
        } else {
            if (data == null) {
                writer.struct(name).writeNull();
            } else {
                writeStruct(writer.struct(name), data, allocator);
            }
        }
    }
}
