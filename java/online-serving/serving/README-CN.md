# MetaSpore 模型 Serving 服务 Java Client 使用方法

Serving 服务的 Java 的客户端基于 GRPC SpringBoot Starter 封装。

## 1. 引入依赖
首先在 Java 工程中引入 Serving Java Client 的依赖。目前还没有把 maven 库发布到公共服务上，可以先 maven install 到本地的 maven 仓库：
```bash
git clone https://github.com/meta-soul/MetaSpore.git
cd MetaSpore/java/online-serving/serving
mvn install -DskipTests -Dmaven.test.skip=true
```

## 2. 在项目中引入依赖
```xml
<dependency>
    <groupId>com.dmetasoul</groupId>
    <artifactId>serving</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

## 3. 调用方法
在 SpringBoot 工程中，需要使用 client 的 controller 中，首先注入 GrpcClient，然后就可以构建 FeatureTable，并调用 serving 服务，参考 [XGBoostController.java](src/test/java/com/dmetasoul/metaspore/serving/XGBoostController.java)：

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

## 4. 测试方法
- 可以写一个 main 函数来启动 Spring boot 并调用自己的 controller 进行测试。如果用 main 方法测试，需要在 resources 目录下添加 application.properties 文件，配置 GRPC 的初始化：
    ```ini
    grpc.client.metaspore.negotiationType=PLAINTEXT
    grpc.client.metaspore.address=static://127.0.0.1:50000
    ```

    这些配置中，metaspore 是上面 Java 代码中 @GrpcClient("metaspore") 注解的服务名。地址可以配成一个静态的测试服务IP。生产环境中通常会配置成服务发现的方式。
- 也可以通过 SpringBootTest 的方法，直接在 JUnit 测试用例中调用 controller：
    ```java
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
    import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

    @ActiveProfiles("test")
    @SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.MOCK,
        classes = {XGBoostController.class, GrpcClientAutoConfiguration.class}
    )
    @AutoConfigureMockMvc
    public class DenseXGBoostTest {

        @Autowired
        private MockMvc mvc;

        @Test
        public void testXGBoostPredict() throws Exception {
            mvc.perform(MockMvcRequestBuilders.get("/xgboost_predict?user_id=xxx").accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isOk())
                    .andExpect(content().string(equalTo("{\"label\":[0.0],\"label_shape\":[1],\"probabilities\":[0.7300644,0.2699356],\"probabilities_shape\":[1,2]}")));
        }
    }
    ```

    注意这里配置了 @ActiveProfiles("test") ，也即这个单元测试使用的是 src/test/resources/application-test.properties 这个配置文件。