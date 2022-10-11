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

import reactor.core.publisher.Mono;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class ServingClient {
    public static Map<String, ArrowTensor> predictBlocking(PredictGrpc.PredictBlockingStub client,
                                                           String modelName,
                                                           Iterable<FeatureTable> featureTables,
                                                           ArrowAllocator allocator,
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
            map.put(name, TensorSerDe.deserializeFrom(name, reply, allocator));
        }
        return map;
    }

    public static Mono<Map<String, ArrowTensor>> predictReactor(ReactorPredictGrpc.ReactorPredictStub client,
                                                                String modelName,
                                                                Iterable<FeatureTable> featureTables,
                                                                ArrowAllocator allocator,
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
                    map.put(name, TensorSerDe.deserializeFrom(name, predictReply, allocator));
                } catch (IOException e) {
                    sink.error(e);
                }
            }
            sink.next(map);
        });
    }
}