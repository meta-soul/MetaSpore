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

package com.dmetasoul.metaspore.serving;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.impl.UnionListWriter;
import org.apache.arrow.vector.holders.VarCharHolder;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.Text;

import java.nio.charset.StandardCharsets;

public class FeatureTable implements AutoCloseable {

    public FeatureTable(String name, Iterable<Field> fields, BufferAllocator allocator) {
        this.name = name;
        this.allocator = new ArrowAllocator(allocator);
        Schema schema = new Schema(fields);
        root = VectorSchemaRoot.create(schema, this.allocator.getAlloc());
    }

    public FeatureTable(String name, Iterable<Field> fields) {
        this.name = name;
        this.allocator = new ArrowAllocator(Integer.MAX_VALUE);
        Schema schema = new Schema(fields);
        root = VectorSchemaRoot.create(schema, allocator.getAlloc());
    }
    @Override
    public void close() {
        if (root != null && allocator != null) {
            root.clear();
            allocator.close();
        }
    }

    public void addBuffer(ArrowBuf buf) {
        if (buf != null) {
            this.allocator.addBuffer(buf);
        }
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

    public void setRowCount(int rowCount) {
        if (this.rowCount < rowCount) {
            this.rowCount = rowCount;
        }
    }

    public int getRowCount() {
        return this.rowCount;
    }

    public void setInt(int index, int value, IntVector v) {
        v.setSafe(index, value);
        setRowCount(index + 1);
    }

    public void setLong(int index, long value, BigIntVector v) {
        v.setSafe(index, value);
        setRowCount(index + 1);
    }

    public void setFloat(int index, float value, Float4Vector v) {
        v.setSafe(index, value);
        setRowCount(index + 1);
    }

    public void setDouble(int index, double value, Float8Vector v) {
        v.setSafe(index, value);
        setRowCount(index + 1);
    }

    public void setString(int index, String value, VarCharVector v) {
        v.setSafe(index, new Text(value));
        setRowCount(index + 1);
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
            this.allocator.addBuffer(vch.buffer);
            writer.write(vch);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void setLongList(int index, Iterable<Long> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (Long l : values) {
            writer.writeBigInt(l);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void setLongList(int index, long[] values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (long l : values) {
            writer.writeBigInt(l);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void setIntList(int index, Iterable<Integer> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (Integer i : values) {
            writer.writeInt(i);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void setIntList(int index, int[] values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (int i : values) {
            writer.writeInt(i);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void setFloatList(int index, Iterable<Float> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (Float f : values) {
            writer.writeFloat4(f);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void setFloatList(int index, float[] values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (float f : values) {
            writer.writeFloat4(f);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void setDoubleList(int index, Iterable<Double> values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (double d : values) {
            writer.writeFloat8(d);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void setDoubleList(int index, double[] values, ListVector v) {
        UnionListWriter writer = v.getWriter();
        writer.setPosition(index);
        writer.startList();
        for (double d : values) {
            writer.writeFloat8(d);
        }
        writer.endList();
        setRowCount(index + 1);
    }

    public void finish() {
        root.setRowCount(rowCount);
    }

    private int rowCount = 0;

    private VectorSchemaRoot root;
    private ArrowAllocator allocator;
    private String name;
}