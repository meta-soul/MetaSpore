# MetaSpore Model Serving Service Java Client Usage

The Java client of the Serving service is based on the GRPC SpringBoot Starter package.

## 1. Import dependencies
First, the dependency of Serving Java Client is introduced into the Java project. At present, the maven library has not been released to the public service, you can first maven install to the local maven repository:
```bash
git clone https://github.com/meta-soul/MetaSpore.git
cd MetaSpore/java/online-serving/serving
mvn install -DskipTests -Dmaven.test.skip=true
````

## 2. Introduce dependencies into the project
````xml
<dependency>
    <groupId>com.dmetasoul.metaspore</groupId>
    <artifactId>serving</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
````

## 3. Call the method
In the SpringBoot project, you need to use the client controller, first inject GrpcClient, then you can build the FeatureTable, and call the serving service, refer to [XGBoostController.java](src/test/java/com/ dmetasoul/metaspore/serving/XGBoostController.java):

```java
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
        FeatureTable userTable = new FeatureTable("input", userFields, ArrowAllocator.getAllocator());
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
                List.of(userTable), Collections.emptyMap());
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
```

## 4. Test method
- You can write a main function to start Spring boot and call your own controller for testing. If you use the main method to test, you need to add the application.properties file in the resources directory to configure the initialization of GRPC:
    ````ini
    grpc.client.metaspore.negotiationType=PLAINTEXT
    grpc.client.metaspore.address=static://172.31.37.47:50000
    ````

    In these configurations, metaspore is the service name annotated with @GrpcClient("metaspore") in the Java code above. The address can be configured as a static test service IP. In a production environment, it is usually configured as a service discovery method.
- You can also call the controller directly in the JUnit test case through the SpringBootTest method:
    ````java
    package com.dmetasoul.metaspore.serving;

    import net.devh.boot.grpc.client.autoconfigure.GrpcClientAutoConfiguration;
    import org.junit.jupiter.api.Test;
    import org.springframework.beans.factory.annotation.Autowired;
    import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
    import org.springframework.boot.test.context.SpringBootTest;
    import org.springframework.http.MediaType;
    import org.springframework.test.context.ActiveProfiles;
    import org.springframework.test.web.servlet.MockMvc;
    import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

    import static org.hamcrest.Matchers.equalTo;
    import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
    import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.stat