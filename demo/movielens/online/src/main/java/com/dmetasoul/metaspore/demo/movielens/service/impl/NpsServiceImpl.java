package com.dmetasoul.metaspore.demo.movielens.service.impl;

import com.dmetasoul.metaspore.demo.movielens.service.NpsService;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.dmetasoul.metaspore.serving.PredictGrpc;
import com.dmetasoul.metaspore.serving.ServingClient;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;

@Service
public class NpsServiceImpl implements NpsService {
    @GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;

    @Override
    public Map<String, ArrowTensor> predictBlocking(String modelName, Iterable<FeatureTable> featureTables, Map<String, String> parameters) throws IOException {
        return ServingClient.predictBlocking(client, modelName, featureTables, parameters);
    }

    @Override
    public List<List<Float>> getVectorsFromNpsResult(Map<String, ArrowTensor> nspResultMap, String targetKey) {
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
    public List<Float> getScoresFromNpsResult(Map<String, ArrowTensor> nspResultMap, String targetKey, int targetIndex) {
        ArrowTensor tensor = nspResultMap.get(targetKey);
        ArrowTensor.FloatTensorAccessor accessor = tensor.getFloatData();
        long[] shape = tensor.getShape();
        if (targetIndex < 0 || targetIndex >= shape.length) {
            throw new IllegalArgumentException("Target index is out of shape scope. targetIndex: " + targetIndex);
        }
        List<Float> scores = new ArrayList<>();
        for (int i = 0; i < shape[0]; i++) {
            scores.add(accessor.get(i, targetIndex));
        }

        return scores;
    }
}
