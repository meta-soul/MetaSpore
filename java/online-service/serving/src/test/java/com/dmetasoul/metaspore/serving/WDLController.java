package com.dmetasoul.metaspore.serving;

import com.fasterxml.jackson.databind.ObjectMapper;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.assertj.core.util.Lists;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.*;

/**
 * Testing a wide & deep model.
 * Train script can be found at https://github.com/meta-soul/metaspore-serving/blob/main/test/serving/sparse_wdl_export_test.py
 * Column name: https://github.com/meta-soul/metaspore-serving/blob/main/test/serving/schema/wdl/column_name_demo.txt
 * Combine schema: https://github.com/meta-soul/metaspore-serving/blob/main/test/serving/schema/wdl/combine_schema_demo.txt
 */
@RestController
public class WDLController {
    @GrpcClient("metaspore")
    private PredictGrpc.PredictBlockingStub client;

    @GetMapping("/wdl_predict")
    public String predict(@RequestParam(value = "user_id") String userId) throws IOException {
        // prepare prediction request
        List<Field> tableFields = Lists.newArrayList();
        for (int i = 1; i <= 13; i++) {
            tableFields.add(Field.nullablePrimitive("integer_feature_" + i, new ArrowType.Utf8()));
        }
        for (int i = 1; i <= 26; i++) {
            // for List<String> arrow array, the children name must be "item"
            tableFields.add(new Field("categorical_feature_" + i,
                    FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.nullable("item", new ArrowType.Utf8()))));
        }

        FeatureTable lrLayerTable = new FeatureTable("lr_layer", tableFields, ArrowAllocator.getAllocator());
        FeatureTable sparseLayerTable = new FeatureTable("_sparse", tableFields, ArrowAllocator.getAllocator());

        // first line from criteo dataset day_0_0.001_test.csv
        String line = "4\t41\t4\t4\t\t2\t0\t90\t5\t2\t\t1068\t4\ta5ba1c3d\tb292f1dd\ta3c8e366\t386c49ee\t664ff944\t6fcd6dcb\t2f0d9894\t7875e132\t54fc547f\tac062eaf\t750506a2\t5c4adbfa\tbf78d0d4\t\t4f36b1c8\t\t\tb8170bba\t9512c20b\t080347b3\t8e01df1e\t607fc1a8\t\t407e8c65\t337b81aa\t6c730e3e";
        String[] dataFields = line.split("\t");
        if (dataFields.length != 39) {
            return "\"fields length is " + dataFields.length + "\"";
        }
        for (int i = 0; i < 13; ++i) {
            lrLayerTable.setString(0, dataFields[i], lrLayerTable.getVector(i));
            sparseLayerTable.setString(0, dataFields[i], lrLayerTable.getVector(i));
        }
        for (int i = 13; i < 26; ++i) {
            lrLayerTable.setStringList(0, Arrays.asList(dataFields[i]), lrLayerTable.getVector(i));
            sparseLayerTable.setStringList(0, Arrays.asList(dataFields[i]), lrLayerTable.getVector(i));
        }

        // predict and get result tensor
        String modelName = "wide_and_deep";
        Map<String, ArrowTensor> result = ServingClient.predictBlocking(client, modelName,
                List.of(lrLayerTable, sparseLayerTable), new ArrowAllocator(ArrowAllocator.getAllocator()), Collections.emptyMap());

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
