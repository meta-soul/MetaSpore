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
import io.milvus.response.SearchResultsWrapper;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.assertj.core.util.Lists;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;
import java.util.*;

@SpringBootTest
public class MovielensRecommendONNXLightGBMTests {
    @GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;

    private String modelName = "lightgbm_test_model";

    private long userId = 0l;
    private double[] inputFeatures = new double [] {
            0.4156825696740671,3.280314960629921,5.19295685089021,2.8555555555555556,
            0.2777777777777778,9.138951775888529,3.555579422188809,0.5671725145256736
    };

    @Test
    void contextLoads() {
    }

    @Test
    public void testQueryNPSServiceByUserId() throws IOException {
        System.out.println("Test query NPS Service by userId: " + userId);
        String npsResult = predict(String.valueOf(userId));
        System.out.println("Test query NPS Service result: " + npsResult);
    }
    public String predict(String userId) throws IOException {
        // prepare prediction request
        List<Field> userFields = Lists.newArrayList();
        for (int i = 0; i < inputFeatures.length; i++) {
            userFields.add(Field.nullablePrimitive("field_" + i, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)));
        }

        FeatureTable userTable = new FeatureTable("input", userFields, ArrowAllocator.getAllocator());
        for (int i = 0; i < inputFeatures.length; i++) {
            userTable.setFloat(0, (float) inputFeatures[i], userTable.getVector(i));
        }
        for (int i = 0; i < inputFeatures.length; i++) {
            userTable.setFloat(1, (float) inputFeatures[i], userTable.getVector(i));
        }
        for (int i = 0; i < inputFeatures.length; i++) {
            userTable.setFloat(2, (float) inputFeatures[i], userTable.getVector(i));
        }

        // predict and get result tensor
        Map<String, ArrowTensor> result = ServingClient.predictBlocking(client, modelName, Lists.newArrayList(userTable), Collections.emptyMap());

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