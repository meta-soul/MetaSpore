package com.dmetasoul.metaspore.serving;

import reactor.core.publisher.Mono;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class ServingClient {
    public static Map<String, ArrowTensor> predictBlocking(PredictGrpc.PredictBlockingStub client,
                                              String modelName,
                                              Iterable<FeatureTable> featureTables,
                                              Map<String, String> parameters) throws IOException {
        PredictRequest.Builder builder = PredictRequest.newBuilder();
        builder.setModelName(modelName);
        for (FeatureTable table : featureTables) {
            FeatureTableSerDe.serializeTo(table, builder);
        }
        builder.putAllParameters(parameters);
        PredictReply reply = client.predict(builder.build());
        Map<String, ArrowTensor> map = new HashMap<>();
        for (String name : reply.getPayloadMap().keySet()) {
            map.put(name, TensorSerDe.deserializeFrom(name, reply));
        }
        return map;
    }

    public static Mono<Map<String, ArrowTensor>> predictReactor(ReactorPredictGrpc.ReactorPredictStub client,
                                                                String modelName,
                                                                Iterable<FeatureTable> featureTables,
                                                                Map<String, String> parameters) throws IOException {
        PredictRequest.Builder builder = PredictRequest.newBuilder();
        builder.setModelName(modelName);
        for (FeatureTable table : featureTables) {
            FeatureTableSerDe.serializeTo(table, builder);
        }
        builder.putAllParameters(parameters);
        return client.predict(builder.build()).handle((predictReply, sink) -> {
            Map<String, ArrowTensor> map = new HashMap<>();
            for (String name : predictReply.getPayloadMap().keySet()) {
                try {
                    map.put(name, TensorSerDe.deserializeFrom(name, predictReply));
                } catch (IOException e) {
                    sink.error(e);
                }
            }
            sink.next(map);
        });
    }
}
