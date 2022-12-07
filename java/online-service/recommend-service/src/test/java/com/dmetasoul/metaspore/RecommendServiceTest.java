package com.dmetasoul.metaspore;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@ActiveProfiles("test")
@SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.MOCK,
        args = {
                "--init_config=../../../../bigdata-flow-sagemaker-test/online-volume/recommend-config.yaml",
                "--init_config_format=yaml",
                "--init_model_info=../../../../bigdata-flow-sagemaker-test/online-volume/model-infos.json"
        }
)
@AutoConfigureMockMvc
public class RecommendServiceTest {
    @Autowired
    private MockMvc mvc;

    @Test
    public void testInvocation() throws Exception {
        System.out.println(System.getProperty("user.dir"));
        mvc.perform(MockMvcRequestBuilders.post("/invocations")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"operator\": \"recommend\", \"request\": {\"user_id\": \"A2DU7MTSGFQ0D3\", \"scene\": \"guess-you-like\"}}")
                        .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andDo(result -> System.out.println(result.getResponse().getContentAsString()));
    }
}
