package com.dmetasoul.metaspore.demo.multimodal;

import com.dmetasoul.metaspore.demo.multimodal.service.impl.HfPreprocessorServiceImpl;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.google.protobuf.ByteString;
import org.assertj.core.util.Lists;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;

import java.util.*;
import java.io.IOException;

@SpringBootTest
public class MultiModalRetrievalHfPreprocessorTests {
    @Autowired
    private ApplicationContext context;

    @Test
    void contextLoads() {
    }

    @Test
    public void testTokenizerHfPreprocessor() {
        System.out.println("Test HuggingFace tokenizer:");
        String modelName = "sbert-chinese-qmc-domain-v1";

        //HfPreprocessorServiceImpl service = new HfPreprocessorServiceImpl("127.0.0.1", 60051);
        HfPreprocessorServiceImpl service = context.getBean(HfPreprocessorServiceImpl.class);

        List<String> texts = new ArrayList<>();
        texts.add("hello world!");
        texts.add("我在中国北京");

        Map<String, ByteString> serviceResults = new HashMap<>();
        Map<String, ArrowTensor> results = new HashMap<>();
        try {
            serviceResults = service.predictBlocking(modelName, texts, Collections.emptyMap());
            results = service.pbToArrow(serviceResults);
        } catch (IOException e) {
            System.out.println("Exception: " + e.getMessage());
            return;
        }

        System.out.println("Results: ");
        System.out.println(service.getIntPredictFromArrowTensorResult(results));
    }
}
