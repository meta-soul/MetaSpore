package com.dmetasoul.metaspore.demo.movielens;

import com.dmetasoul.metaspore.serving.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.assertj.core.util.Lists;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;
import java.util.*;

@SpringBootTest
public class MovielensRecommendWideAndDeepTests {
    @GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;

    private String modelName = "wide_and_deep";

    private long userId = 10l;

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
        List<Field> tableFields = Lists.newArrayList();
        for (int i = 1; i <= 13; i++) {
            tableFields.add(Field.nullablePrimitive("integer_feature_" + i, new ArrowType.Utf8()));
        }
        for (int i = 1; i <= 26; i++) {
            tableFields.add(Field.nullablePrimitive("categorical_feature_" + i, new ArrowType.Utf8()));
        }

        FeatureTable lrLayerTable = new FeatureTable("lr_layer", tableFields, ArrowAllocator.getAllocator());
        FeatureTable sparseLayerTable = new FeatureTable("_sparse", tableFields, ArrowAllocator.getAllocator());

        // first line from criteo dataset day_0_0.001_test.csv
        String line = "4\t41\t4\t4\t\t2\t0\t90\t5\t2\t\t1068\t4\ta5ba1c3d\tb292f1dd\ta3c8e366\t386c49ee\t664ff944\t6fcd6dcb\t2f0d9894\t7875e132\t54fc547f\tac062eaf\t750506a2\t5c4adbfa\tbf78d0d4\t\t4f36b1c8\t\t\tb8170bba\t9512c20b\t080347b3\t8e01df1e\t607fc1a8\t\t407e8c65\t337b81aa\t6c730e3e";
        String[] dataFields = line.split("\t");
        if (dataFields.length != 39) {
            return "\"fields length is " + dataFields.length + "\"";
        }
        for (int i = 0; i < 39; ++i) {
            lrLayerTable.setString(0, dataFields[i], lrLayerTable.getVector(i));
            sparseLayerTable.setString(0, dataFields[i], lrLayerTable.getVector(i));
        }

        // predict and get result tensor
        Map<String, ArrowTensor> result = ServingClient.predictBlocking(client, modelName,
                List.of(lrLayerTable, sparseLayerTable), Collections.emptyMap());

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
