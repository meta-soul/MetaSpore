package com.dmetasoul.metaspore.serving;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class ArrowTensorSerdeTest {
    static public void main(String[] args) throws IOException {
        float[] floatData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        long[] floatShape = {2L, 3L};
        ArrowTensor floatTensor = ArrowTensor.createFromFloatArray(floatShape, floatData, null, null);

        double[] doubleData = {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};
        long[] doubleShape = {2L, 2L, 2L};
        ArrowTensor doubleTensor = ArrowTensor.createFromDoubleArray(doubleShape, doubleData, null, null);

        int[] intData = {111, 222};
        long[] intShape = {1L, 2L};
        ArrowTensor intTensor = ArrowTensor.createFromIntArray(intShape, intData, null, null);

        long[] longData = {333L, 444L};
        long[] longShape = {2L, 1L};
        ArrowTensor longTensor = ArrowTensor.createFromLongArray(longShape, longData, null, null);

        PredictRequest.Builder builder = PredictRequest.newBuilder();
        TensorSerDe.serializeTo("float_tensor", floatTensor, builder);
        TensorSerDe.serializeTo("double_tensor", doubleTensor, builder);
        TensorSerDe.serializeTo("int_tensor", intTensor, builder);
        TensorSerDe.serializeTo("long_tensor", longTensor, builder);

        PredictRequest request = builder.build();

        OutputStream os = new BufferedOutputStream(new FileOutputStream("arrow_tensor_java_ser.bin"));
        request.writeTo(os);
        os.close();
    }
}
