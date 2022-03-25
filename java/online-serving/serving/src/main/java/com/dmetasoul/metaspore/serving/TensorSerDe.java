package com.dmetasoul.metaspore.serving;

import org.apache.arrow.flatbuf.Tensor;

import java.io.IOException;

public class TensorSerDe {
    public static void serializeTo(String name, Tensor tensor, PredictRequest request) {
    }

    public static ArrowTensor deserializeFrom(String name, PredictReply response) throws IOException {
        return ArrowTensor.readFromByteString(response.getPayloadOrThrow(name));
    }
}
