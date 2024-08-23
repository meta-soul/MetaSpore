package com.dmetasoul.metaspore.serving;

import com.fasterxml.jackson.databind.ObjectMapper;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.*;

@RestController
public class XGBoostController {
    @GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;

    @GetMapping("/xgboost_predict")
    public String predict(@RequestParam(value = "user_id") String userId) throws IOException {
        List<Field> userFields = new ArrayList<>();
        for (int i = 0; i < 10; ++i) {
            userFields.add(Field.nullablePrimitive("field_" + i, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)));
        }
        ArrowAllocator allocator = new ArrowAllocator("predict", Long.MAX_VALUE);
        FeatureTable userTable = new FeatureTable("input", userFields, allocator);
        userTable.setFloat(0, 0.6558618f, userTable.getVector(0));
        userTable.setFloat(0, 0.13005558f, userTable.getVector(1));
        userTable.setFloat(0, 0.03510657f, userTable.getVector(2));
        userTable.setFloat(0, 0.23048967f, userTable.getVector(3));
        userTable.setFloat(0, 0.63329154f, userTable.getVector(4));
        userTable.setFloat(0, 0.43201634f, userTable.getVector(5));
        userTable.setFloat(0, 0.5795548f, userTable.getVector(6));
        userTable.setFloat(0, 0.5384891f, userTable.getVector(7));
        userTable.setFloat(0, 0.9612295f, userTable.getVector(8));
        userTable.setFloat(0, 0.39274803f, userTable.getVector(9));

        // predict and get result tensor
        Map<String, ArrowTensor> result = ServingClient.predictBlocking(client, "xgboost_model",
                List.of(userTable), allocator, Collections.emptyMap());
        Map<String, Object> toJson = new TreeMap<>();
        for (Map.Entry<String, ArrowTensor> entry : result.entrySet()) {
            ArrowTensor tensor = result.get(entry.getKey());
            long[] shape = tensor.getShape();
            toJson.put(entry.getKey() + "_shape", shape);
            if (tensor.isFloatTensor()) {
                ArrowTensor.FloatTensorAccessor accessor = tensor.getFloatData();
                long eleNum = tensor.getSize();
                List<Float> l = new ArrayList<>();
                for (int i = 0; i < (int) eleNum; ++i) {
                    l.add(accessor.get(i));
                }
                toJson.put(entry.getKey(), l);
            } else if (tensor.isLongTensor()) {
                ArrowTensor.LongTensorAccessor accessor = tensor.getLongData();
                long eleNum = tensor.getSize();
                List<Long> l = new ArrayList<>();
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
