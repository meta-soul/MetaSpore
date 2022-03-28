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

package com.dmetasoul.metaspore.demo.movielens;

import com.dmetasoul.metaspore.serving.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.assertj.core.util.Lists;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;
import java.util.*;

@SpringBootTest
public class MovielensRecommendTwoTowersTests {
    @GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;

    private String modelName = "two_towers_simplex";

    private String userId = "10";

    private List<String> recentMovieIds = List.of("3593", "275", "3034", "3447", "595", "923", "1704", "1283", "2693", "1221");

    @Test
    void contextLoads() {
    }

    @Test
    public void testQueryNPSServiceByUserId() throws IOException {
        System.out.println("Test query NPS Service by userId: " + userId);
        String npsResult = predict(userId, recentMovieIds);
        System.out.println("Test query NPS Service result: " + npsResult);
    }

    public String predict(String userId, List<String> recentMovieIds) throws IOException {
        // prepare prediction request
        List<Field> userFields = List.of(
                Field.nullablePrimitive("user_id", ArrowType.Utf8.INSTANCE)
        );

        List<Field> interactedItemsFields = List.of(
                new Field("recent_movie_ids", FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.nullable("item", new ArrowType.Utf8())))
        );

        FeatureTable userTable = new FeatureTable("_sparse_user", userFields, ArrowAllocator.getAllocator());
        FeatureTable interactedItemsTable = new FeatureTable("_sparse_interacted_items", interactedItemsFields, ArrowAllocator.getAllocator());

        userTable.setString(0, userId, userTable.getVector(0));
        interactedItemsTable.setStringList(0, recentMovieIds, interactedItemsTable.getVector(0));

        // predict and get result tensor
        Map<String, ArrowTensor> result = ServingClient.predictBlocking(client, modelName,
                List.of(userTable, interactedItemsTable), Collections.emptyMap());

        // parse the result tensor
        Map<String, Object> toJson = new TreeMap<>();
        for (Map.Entry<String, ArrowTensor> entry : result.entrySet()) {
            ArrowTensor tensor = result.get(entry.getKey());
            long[] shape = tensor.getShape();
            toJson.put(entry.getKey() + "_shape", shape);
            ArrowTensor.FloatTensorAccessor accessor = result.get(entry.getKey()).getFloatData();
            long eleNum = tensor.getSize();
            if (accessor != null) {
                List<Float> l = new ArrayList<>();
                for (int i = 0; i < (int) eleNum; ++i) {
                    l.add(accessor.get(i));
                }
                toJson.put(entry.getKey(), l);
            } else {
                toJson.put(entry.getKey(), null);
            }
        }
        return new ObjectMapper().writeValueAsString(toJson);
    }
}