package com.dmetasoul.metaspore.serving;

import com.fasterxml.jackson.databind.ObjectMapper;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.assertj.core.util.Lists;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.*;

@RestController
public class LightGBMController {
    @GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;

    private String modelName = "lightgbm_test_model";


    private double[] inputFeatures = new double[]{
            0.4156825696740671, 3.280314960629921, 5.19295685089021, 2.8555555555555556,
            0.2777777777777778, 9.138951775888529, 3.555579422188809, 0.5671725145256736
    };

    @GetMapping("/lightgbm_predict")
    public String predict(@RequestParam(value = "user_id") String userId) throws IOException {
        // prepare prediction request
        List<Field> userFields = Lists.newArrayList();
        for (int i = 0; i < inputFeatures.length; i++) {
            userFields.add(Field.nullablePrimitive("field_" + i, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)));
        }

        try (
            ArrowAllocator allocator = new ArrowAllocator("predict", Long.MAX_VALUE);
            FeatureTable userTable = new FeatureTable("input", userFields, allocator);
        ) {
            for (int i = 0; i < inputFeatures.length; i++) {
                userTable.setFloat(0, (float) inputFeatures[i], userTable.getVector(i));
            }

            // predict and get result tensor
            Map<String, ArrowTensor> result = ServingClient.predictBlocking(client, modelName, List.of(userTable), allocator, Collections.emptyMap());

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
}
