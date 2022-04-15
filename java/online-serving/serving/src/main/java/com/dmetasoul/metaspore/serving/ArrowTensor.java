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

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.protobuf.ByteString;
import org.apache.arrow.flatbuf.*;
import org.apache.arrow.vector.ipc.WriteChannel;
import org.apache.arrow.vector.ipc.message.IpcOption;
import org.apache.arrow.vector.ipc.message.MessageSerializer;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;

import java.io.IOException;
import java.nio.*;
import java.nio.channels.Channels;

/*
 * A wrapper class for Arrow's Tensor.
 * Provides methods to create, access and (de)serialize Tensor.
 */
public class ArrowTensor {

    public static ArrowTensor readFromByteString(ByteString bs) throws IOException {
        ArrowMessage m = ArrowMessage.readFromByteString(bs);
        if (m.message.headerType() == MessageHeader.Tensor) {
            Tensor tensor = (Tensor) m.message.header(new Tensor());
            if (m.body != null) {
                return new ArrowTensor(tensor, m.body.nioBuffer());
            } else {
                return new ArrowTensor(tensor, null);
            }
        }
        return null;
    }

    public static ByteString writeToByteString(ArrowTensor tensor) throws IOException {
        FlatBufferBuilder builder = new FlatBufferBuilder();

        ByteBuffer data = tensor.data;

        // build Tensor table
        int typeOffset = 0;
        Runnable addTypeTypeFunc = null;
        if (tensor.isDoubleTensor()) {
            addTypeTypeFunc = () -> Tensor.addTypeType(builder, Type.FloatingPoint);
            typeOffset = FloatingPoint.createFloatingPoint(builder, Precision.DOUBLE);
            if (data == null) {
                data = ByteBuffer.allocate(tensor.doubleData.capacity() * 8);
                data.order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().put(tensor.doubleData);
            }
        } else if (tensor.isFloatTensor()) {
            addTypeTypeFunc = () -> Tensor.addTypeType(builder, Type.FloatingPoint);
            typeOffset = FloatingPoint.createFloatingPoint(builder, Precision.SINGLE);
            if (data == null) {
                data = ByteBuffer.allocate(tensor.floatData.capacity() * 4);
                data.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(tensor.floatData);
            }
        } else if (tensor.isIntegerTensor()) {
            addTypeTypeFunc = () -> Tensor.addTypeType(builder, Type.Int);
            typeOffset = Int.createInt(builder, 32, true);
            if (data == null) {
                data = ByteBuffer.allocate(tensor.intData.capacity() * 4);
                data.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().put(tensor.intData);
            }
        } else if (tensor.isLongTensor()) {
            addTypeTypeFunc = () -> Tensor.addTypeType(builder, Type.Int);
            typeOffset = Int.createInt(builder, 64, true);
            if (data == null) {
                data = ByteBuffer.allocate(tensor.longData.capacity() * 8);
                data.order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().put(tensor.longData);
            }
        }

        // create shape vector of tensordim
        int[] shapeVectorOffsets = new int[tensor.shape.length];
        for (int i = 0; i < tensor.shape.length; ++i) {
            long dim = tensor.shape[i];
            String name = "";
            if (tensor.names != null) {
                name = tensor.names[i];
            }
            shapeVectorOffsets[i] = TensorDim.createTensorDim(builder, dim, builder.createString(name));
        }
        
        int shapeVector = Tensor.createShapeVector(builder, shapeVectorOffsets);

        int stridesOffset = 0;
        if (tensor.strides != null) {
            stridesOffset = Tensor.createStridesVector(builder, tensor.strides);
        }

        long bodyLength = tensor.dataElems * tensor.elemBytes;

        Tensor.startTensor(builder);
        Tensor.addType(builder, typeOffset);
        addTypeTypeFunc.run();
        Tensor.addShape(builder, shapeVector);
        // create strides vector
        if (tensor.strides != null) {
            Tensor.addStrides(builder, stridesOffset);
        }
        int bufferOffset = org.apache.arrow.flatbuf.Buffer.createBuffer(builder, 0, bodyLength);
        Tensor.addData(builder, bufferOffset);
        int tensorTableOffset = Tensor.endTensor(builder);

        // build Message Table and get ByteBuffer
        ByteBuffer messageHeader = MessageSerializer.serializeMessage(
            builder, MessageHeader.Tensor, tensorTableOffset, bodyLength, IpcOption.DEFAULT);

        // we need to use MessageSerializer to get final ByteString with 8-byte padding
        ByteString.Output headerOut = ByteString.newOutput();
        WriteChannel out = new WriteChannel(Channels.newChannel(headerOut));
        MessageSerializer.writeMessageBuffer(out, messageHeader.remaining(), messageHeader);
        ByteString bodyBS = ByteString.copyFrom(data);
        ByteString headerBS = headerOut.toByteString();

        // concat body ByteString
        return headerBS.concat(bodyBS);
    }

    public static ArrowTensor createFromFloatArray(long[] shape, float[] data, String[] dimNames, long[] strides) {
        ArrowTensor tensor = new ArrowTensor(shape, dimNames, strides);
        tensor.floatData= FloatBuffer.wrap(data);
        tensor.dataElems = data.length;
        tensor.elemBytes = 4;
        tensor.type = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
        return tensor;
    }

    public static ArrowTensor createFromDoubleArray(long[] shape, double[] data, String[] dimNames, long[] strides) {
        ArrowTensor tensor = new ArrowTensor(shape, dimNames, strides);
        tensor.doubleData= DoubleBuffer.wrap(data);
        tensor.dataElems = data.length;
        tensor.elemBytes = 8;
        tensor.type = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
        return tensor;
    }

    public static ArrowTensor createFromIntArray(long[] shape, int[] data, String[] dimNames, long[] strides) {
        ArrowTensor tensor = new ArrowTensor(shape, dimNames, strides);
        tensor.intData= IntBuffer.wrap(data);
        tensor.dataElems = data.length;
        tensor.elemBytes = 4;
        tensor.type = new ArrowType.Int(32, true);
        return tensor;
    }

    public static ArrowTensor createFromLongArray(long[] shape, long[] data, String[] dimNames, long[] strides) {
        ArrowTensor tensor = new ArrowTensor(shape, dimNames, strides);
        tensor.longData= LongBuffer.wrap(data);
        tensor.dataElems = data.length;
        tensor.elemBytes = 8;
        tensor.type = new ArrowType.Int(64, true);
        return tensor;
    }

    public ArrowTensor(long[] shape, String[] dimNames, long[] strides) {
        this.shape = shape;
        this.names = dimNames;
        this.strides = strides;
    }

    public ArrowTensor(Tensor tensor, ByteBuffer data) {
        int dims = tensor.shapeLength();
        this.shape = new long[dims];
        this.names = new String[dims];
        for (int i = 0; i < dims; ++i) {
            org.apache.arrow.flatbuf.TensorDim dim = tensor.shape(i);
            shape[i] = dim.size();
            names[i] = dim.name();
        }
        int strides_n = tensor.stridesLength();
        strides = new long[strides_n];
        for (int i = 0; i < strides_n; ++i) {
            strides[i] = tensor.strides(i);
        }
        this.type = getTensorArrowType(tensor);
        this.data = data;
        setTypedData();
    }

    public void setTypedData() {
        if (isDoubleTensor()) {
            this.doubleData = data.order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
            this.dataElems = this.doubleData.array().length;
        } else if (isFloatTensor()) {
            this.floatData = data.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            this.dataElems = this.floatData.array().length;
        } else if (isIntegerTensor()) {
            this.intData = data.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
            this.dataElems = this.intData.array().length;
        } else if (isLongTensor()) {
            this.longData = data.order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
            this.dataElems = this.longData.array().length;
        }
    }

    public static ArrowType getTensorArrowType(Tensor tensor) {
        byte typeType = tensor.typeType();
        switch (typeType) {
            case Type.Int:
                Int i = (Int) tensor.type(new Int());
                return new ArrowType.Int(i.bitWidth(), i.isSigned());
            case Type.FloatingPoint:
                FloatingPoint f = (FloatingPoint) tensor.type(new FloatingPoint());
                return new ArrowType.FloatingPoint(FloatingPointPrecision.fromFlatbufID(f.precision()));
            default:
                return null;
        }
    }

    public boolean isFloatTensor() {
        if (getType() instanceof ArrowType.FloatingPoint) {
            ArrowType.FloatingPoint fpType = (ArrowType.FloatingPoint) getType();
            return fpType.getPrecision() == FloatingPointPrecision.SINGLE;
        }
        return false;
    }

    public boolean isDoubleTensor() {
        if (getType() instanceof ArrowType.FloatingPoint) {
            ArrowType.FloatingPoint fpType = (ArrowType.FloatingPoint) getType();
            return fpType.getPrecision() == FloatingPointPrecision.DOUBLE;
        }
        return false;
    }

    public boolean isLongTensor() {
        if (getType() instanceof ArrowType.Int) {
            ArrowType.Int intType = (ArrowType.Int) getType();
            return intType.getBitWidth() == 64;
        }
        return false;
    }

    public boolean isIntegerTensor() {
        if (getType() instanceof ArrowType.Int) {
            ArrowType.Int intType = (ArrowType.Int) getType();
            return intType.getBitWidth() == 32;
        }
        return false;
    }

    public ArrowType getType() {
        return type;
    }

    public long[] getShape() {
        return shape;
    }

    public String[] getNames() {
        return names;
    }

    public long[] getStrides() {
        return strides;
    }

    public ByteBuffer getData() {
        return data;
    }

    public long getSize() {
        if (data == null) {
            return 0;
        }
        long prod = 1;
        for (long l : shape) {
            prod *= l;
        }
        return prod;
    }

    static public class FloatTensorAccessor {

        FloatTensorAccessor(FloatBuffer buffer, int cols) {
            this.buffer = buffer;
            this.cols = cols;
        }

        public float get(int index) {
            return buffer.get(index);
        }

        public float get(int row, int col) {
            return buffer.get(row * cols + col);
        }

        private final int cols;

        private final FloatBuffer buffer;
    }

    static public class DoubleTensorAccessor {

        DoubleTensorAccessor(DoubleBuffer buffer, int cols) {
            this.buffer = buffer;
            this.cols = cols;
        }

        public double get(int index) {
            return buffer.get(index);
        }

        public double get(int row, int col) {
            return buffer.get(row * cols + col);
        }

        private final int cols;

        private final DoubleBuffer buffer;
    }

    static public class IntTensorAccessor {

        IntTensorAccessor(IntBuffer buffer, int cols) {
            this.buffer = buffer;
            this.cols = cols;
        }

        public int get(int index) {
            return buffer.get(index);
        }

        public int get(int row, int col) {
            return buffer.get(row * cols + col);
        }

        private final int cols;

        private final IntBuffer buffer;
    }

    static public class LongTensorAccessor {

        LongTensorAccessor(LongBuffer buffer, int cols) {
            this.buffer = buffer;
            this.cols = cols;
        }

        public long get(int index) {
            return buffer.get(index);
        }

        public long get(int row, int col) {
            return buffer.get(row * cols + col);
        }

        private final int cols;

        private final LongBuffer buffer;
    }

    private static final char[] HEX_ARRAY = "0123456789ABCDEF".toCharArray();

    public static String bytesToHex(ByteBuffer bytes) {
        char[] hexChars = new char[bytes.limit() * 2];
        for (int j = 0; j < bytes.limit(); j++) {
            int v = bytes.get(j) & 0xFF;
            hexChars[j * 2] = HEX_ARRAY[v >>> 4];
            hexChars[j * 2 + 1] = HEX_ARRAY[v & 0x0F];
        }
        return new String(hexChars);
    }

    public FloatTensorAccessor getFloatData() {
        if (floatData != null) {
            return new FloatTensorAccessor(floatData, shape.length == 2 ? (int) shape[1] : 0);
        } else {
            return null;
        }
    }

    public IntTensorAccessor getIntData() {
        if (intData != null) {
            return new IntTensorAccessor(intData, shape.length == 2 ? (int) shape[1] : 0);
        } else {
            return null;
        }
    }

    public DoubleTensorAccessor getDoubleData() {
        if (doubleData != null) {
            return new DoubleTensorAccessor(doubleData, shape.length == 2 ? (int) shape[1] : 0);
        } else {
            return null;
        }
    }

    public LongTensorAccessor getLongData() {
        if (longData != null) {
            return new LongTensorAccessor(longData, shape.length == 2 ? (int) shape[1] : 0);
        } else {
            return null;
        }
    }

    private ArrowType type;
    private long[] shape;
    private String[] names;
    private long[] strides;
    private long dataElems;
    private int elemBytes;
    private ByteBuffer data;
    private FloatBuffer floatData;
    private DoubleBuffer doubleData;
    private IntBuffer intData;
    private LongBuffer longData;
}