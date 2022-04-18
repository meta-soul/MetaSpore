package com.dmetasoul.metaspore.demo.multimodal.service.impl;

import com.dmetasoul.metaspore.demo.multimodal.service.HfPreprocessorService;
import com.dmetasoul.metaspore.demo.multimodal.service.HfPreprocessorGrpc;
import com.dmetasoul.metaspore.demo.multimodal.service.HfTokenizerRequest;
import com.dmetasoul.metaspore.demo.multimodal.service.HfTokenizerResponse;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.TensorSerDe;
import com.google.protobuf.ByteString;
import com.alibaba.fastjson.JSON;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.springframework.stereotype.Service;
import org.apache.arrow.flatbuf.Tensor;
import java.util.*;
import java.io.IOException;


@Service
public class HfPreprocessorServiceImpl implements HfPreprocessorService {
    // wrap gRPC by annotator
    @GrpcClient("hf_preprocessor")
    private HfPreprocessorGrpc.HfPreprocessorBlockingStub client;

    /*
    // provide gRPC init variable and method

    //private ManagedChannel channel;
    //private HfPreprocessorGrpc.HfPreprocessorBlockingStub client;

    public HfPreprocessorServiceImpl () {

    }

    public HfPreprocessorServiceImpl(String host, int port) {
        channel = ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext()
                .build();
        client = HfPreprocessorGrpc.newBlockingStub(channel);
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }*/

    @Override
    public Map<String, ByteString> predictBlocking(String modelName, List<String> texts, Map<String, String> parameters) throws IOException {
        Map<String, ByteString> payload = new HashMap<>();
        payload.put("texts", ByteString.copyFrom(JSON.toJSONBytes(texts)));

        HfTokenizerRequest.Builder builder = HfTokenizerRequest.newBuilder();
        builder.setModelName(modelName);
        builder.putAllParameters(Collections.emptyMap());
        builder.putAllPayload(payload);

        HfTokenizerResponse response = client.hfTokenizer(builder.build());

        /*Map<String, ArrowTensor> map = new HashMap<>();
        for (String name : response.getPayloadMap().keySet()) {
            map.put(name, ArrowTensor.readFromByteString(response.getPayloadOrThrow(name)));
        }*/

        Map<String, ByteString> map = new HashMap<>();
        for (String name : response.getPayloadMap().keySet()) {
            map.put(name, response.getPayloadOrThrow(name));
        }

        return map;
    }

    @Override
    public Map<String, ArrowTensor> pbToArrow(Map<String, ByteString> payload) throws IOException {
        Map<String, ArrowTensor> map = new HashMap<>();
        for (String name : payload.keySet()) {
            map.put(name, ArrowTensor.readFromByteString(payload.get(name)));
        }
        return map;
    }

    @Override
    public List<List<Float>> getFloatVectorsFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap, String targetKey) {
        ArrowTensor tensor = nspResultMap.get(targetKey);
        ArrowTensor.FloatTensorAccessor accessor = tensor.getFloatData();
        long[] shape = tensor.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Shape length must equal to 2 (batch, vector dim). shape.length: " + shape.length);
        }
        List<List<Float>> vectors = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            List<Float> vector = new ArrayList<>();
            for (int j = 0; j < shape[1]; j++) {
                vector.add(accessor.get(i, j));
            }
            vectors.add(vector);
        }

        return vectors;
    }

    @Override
    public List<List<Integer>> getIntVectorsFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap, String targetKey) {
        ArrowTensor tensor = nspResultMap.get(targetKey);
        ArrowTensor.IntTensorAccessor accessor = tensor.getIntData();
        long[] shape = tensor.getShape();

        //if (shape.length != 2) {
        //    throw new IllegalArgumentException("Shape length must equal to 2 (batch, vector dim). shape.length: " + shape.length);
        //}

        List<List<Integer>> vectors = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            List<Integer> vector = new ArrayList<>();
            for (int j = 0; j < shape[1]; j++) {
                vector.add(accessor.get(i, j));
            }
            vectors.add(vector);
        }

        return vectors;
    }

    @Override
    public Map<String, List<List<Float>>> getFloatPredictFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap) {
        Map<String, List<List<Float>>> results = new HashMap<>();
        for (String name : nspResultMap.keySet()) {
            results.put(name, getFloatVectorsFromArrowTensorResult(nspResultMap, name));
        }
        return results;
    }

    @Override
    public Map<String, List<List<Integer>>> getIntPredictFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap) {
        Map<String, List<List<Integer>>> results = new HashMap<>();
        for (String name : nspResultMap.keySet()) {
            results.put(name, getIntVectorsFromArrowTensorResult(nspResultMap, name));
        }
        return results;
    }
}
