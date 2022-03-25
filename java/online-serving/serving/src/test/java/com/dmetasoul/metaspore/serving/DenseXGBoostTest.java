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
