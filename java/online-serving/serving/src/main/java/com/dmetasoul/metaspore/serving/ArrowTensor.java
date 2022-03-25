package com.dmetasoul.metaspore.serving;

import com.google.protobuf.ByteString;
import org.apache.arrow.flatbuf.*;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;

import java.io.IOException;
import java.nio.*;

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
        if (data == null) return 0;
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
        if (data != null) {
            return new FloatTensorAccessor(data.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer(), shape.length == 2 ? (int) shape[1] : 0);
        } else
            return null;
    }

    public IntTensorAccessor getIntData() {
        if (data != null)
            return new IntTensorAccessor(data.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer(), shape.length == 2 ? (int) shape[1] : 0);
        else
            return null;
    }

    public DoubleTensorAccessor getDoubleData() {
        if (data != null)
            return new DoubleTensorAccessor(data.order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer(), shape.length == 2 ? (int) shape[1] : 0);
        else
            return null;
    }

    public LongTensorAccessor getLongData() {
        if (data != null)
            return new LongTensorAccessor(data.order(ByteOrder.LITTLE_ENDIAN).asLongBuffer(), shape.length == 2 ? (int) shape[1] : 0);
        else
            return null;
    }

    private final ArrowType type;
    private final long[] shape;
    private final String[] names;
    private final long[] strides;
    private final ByteBuffer data;
}
