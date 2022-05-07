//
// Copyright 2022 DMetaSoul
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.dmetasoul.metaspore.demo.multimodal;

import com.dmetasoul.metaspore.demo.multimodal.service.impl.HfPreprocessorServiceImpl;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.google.protobuf.ByteString;
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
    public void testTokenizerHfPreprocessorModel1() {
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

    @Test
    public void testTokenizerHfPreprocessorModel2() {
        System.out.println("Test HuggingFace tokenizer:");
        String modelName = "clip-text-encoder-v1";

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
