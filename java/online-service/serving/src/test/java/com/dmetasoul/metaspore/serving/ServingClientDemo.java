package com.dmetasoul.metaspore.serving;

import net.devh.boot.grpc.client.inject.GrpcClient;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@RestController
public class ServingClientDemo {
    @GrpcClient("metaspore-serving")
    private PredictGrpc.PredictBlockingStub client;

    @GetMapping("/movielens_wdl")
    public String predict(@RequestParam(value = "user_id") String userId) throws IOException {
        List<Field> userFields = Arrays.asList(
                Field.nullable("user_id", ArrowType.Utf8.INSTANCE)
                , Field.nullable("age", new ArrowType.Int(32, true)) // continuous feature
                // multi-value categorical feature
                , new Field("history_ids",
                        FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.nullable("item", new ArrowType.Utf8())))
        );
        ArrowAllocator allocator = new ArrowAllocator("predict", Long.MAX_VALUE);
        FeatureTable userTable = new FeatureTable("user", userFields, allocator);

        // fill user features
        for (int i = 0; i < 2; ++i) {
            // set string value for first feature of string type
            userTable.setString(i, "user_id_" + i, userTable.getVector(0));
            userTable.setInt(i, i, userTable.getVector(1));
            // set string list value for history_ids feature of list type
            userTable.setStringList(i, Arrays.asList("item0", "item1"), userTable.getVector("history_ids"));
        }

        List<Field> itemFields = Arrays.asList(
                Field.nullable("item_id", ArrowType.Utf8.INSTANCE)
                , Field.nullable("user_id", ArrowType.Utf8.INSTANCE) // reference user id
                , Field.nullable("item_price", new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE))
                , Field.nullable("item_type", ArrowType.Utf8.INSTANCE) // pass string directly for categorical feature
        );
        // fill item features
        FeatureTable itemTable = new FeatureTable("item", itemFields, allocator);
        for (int i = 0; i < 4; ++i) {
            itemTable.setString(i, "item_id_" + i, itemTable.getVector(0));
            itemTable.setString(i, "user_id_" + i % 2, itemTable.getVector(1));
            itemTable.setFloat(i, (float) i * 1.7f, itemTable.getVector("item_price"));
            itemTable.setString(i, "type_" + i, itemTable.getVector("item_type"));
        }

        // predict and get result tensor
        Map<String, ArrowTensor> result = ServingClient.predictBlocking(client, "movielens_wide_and_deep",
                Arrays.asList(userTable, itemTable), allocator, Collections.emptyMap());
        ArrowTensor.FloatTensorAccessor score = result.get("score").getFloatData();
        return "";
    }
}
