package com.dmetasoul.metaspore.serving;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.impl.UnionListWriter;
import org.apache.arrow.vector.holders.VarCharHolder;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.Text;

import java.nio.charset.StandardCharsets;

public class FeatureTable {

    public FeatureTable(String name, Iterable<Field> fields, BufferAllocator allocator) {
        this.name = name;
        Schema schema = new Schema(fields);
        root = VectorSchemaRoot.create(schema, allocator);
    }

    public FeatureTable(String name, VectorSchemaRoot root) {
        this.name = name;
        this.root = root;
    }

    public Schema getSchema() {
        return root.getSchema();
    }

    public String getName() {
        return name;
    }

    public VectorSchemaRoot getRoot() {
        return root;
    }

    public <T extends FieldVector> T getVector(String name) {
        return (T) root.getVector(name);
    }

    public <T extends FieldVector> T getVector(int i) {
        return (T) root.getVector(i);
    }

    public void setInt(int index, int value, IntVector v) {
        v.setSafe(index, value);
        rowCount = index + 1;
    }

    public void setLong(int index, long value, BigIntVector v) {
        v.setSafe(index, value);
        rowCount = index + 1;
    }

    public void setFloat(int index, float value, Float4Vector v) {
        v.setSafe(index, value);
        rowCount = index + 1;
    }

    public void setDouble(int index, double value, Float8Vector v) {
        v.setSafe(index, value);
        rowCount = index + 1;
    }

    public void setString(int index, String value, VarCharVector v) {
        v.setSafe(index, new Text(value));
        rowCount = index + 1;
    }

    public void setStringList(int index, Iterable<String> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (String s : values) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            VarCharHolder vch = new VarCharHolder();
            vch.start = 0;
            vch.end = b.length;
            vch.buffer = v.getAllocator().buffer(b.length);
            vch.buffer.setBytes(0, b);
            writer.write(vch);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void setLongList(int index, Iterable<Long> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (Long l : values) {
            writer.writeBigInt(l);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void setLongList(int index, long[] values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (long l : values) {
            writer.writeBigInt(l);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void setIntList(int index, Iterable<Integer> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (Integer i : values) {
            writer.writeInt(i);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void setIntList(int index, int[] values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (int i : values) {
            writer.writeInt(i);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void setFloatList(int index, Iterable<Float> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (Float f : values) {
            writer.writeFloat4(f);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void setFloatList(int index, float[] values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (float f : values) {
            writer.writeFloat4(f);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void setDoubleList(int index, Iterable<Double> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (double d : values) {
            writer.writeFloat8(d);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void setDoubleList(int index, double[] values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (double d : values) {
            writer.writeFloat8(d);
        }
        writer.endList();
        rowCount = index + 1;
    }

    public void finish() {
        root.setRowCount(rowCount);
    }

    private int rowCount = 0;

    private VectorSchemaRoot root;

    private String name;
}
