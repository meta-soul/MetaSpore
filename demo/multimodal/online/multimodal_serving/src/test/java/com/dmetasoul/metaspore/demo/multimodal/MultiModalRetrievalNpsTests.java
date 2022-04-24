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
import com.dmetasoul.metaspore.demo.multimodal.service.impl.NpsServiceImpl;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.google.protobuf.ByteString;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;

import java.io.IOException;
import java.util.*;

@SpringBootTest
public class MultiModalRetrievalNpsTests {
    @Autowired
    private ApplicationContext context;

    @Test
    void contextLoads() {
    }

    @Test
    public void testHfPreprocessorWithNps() {
        System.out.println("Test HuggingFace tokenizer and MetaSpore NPS:");
        String modelName = "sbert-chinese-qmc-domain-v1";

        HfPreprocessorServiceImpl serviceHf = context.getBean(HfPreprocessorServiceImpl.class);
        NpsServiceImpl serviceNps = context.getBean(NpsServiceImpl.class);

        List<String> texts = new ArrayList<>();
        texts.add("hello world!");
        texts.add("我在中国北京");

        System.out.println("Input: "+texts.toString());

        Map<String, ByteString> serviceResults = new HashMap<>();
        Map<String, ArrowTensor> arrowResults = new HashMap<>();
        try {
            serviceResults = serviceHf.predictBlocking(modelName, texts, Collections.emptyMap());
            arrowResults = serviceHf.pbToArrow(serviceResults);
        } catch (IOException e) {
            System.out.println("Hf-Preprocessor Exception: " + e.getMessage());
            return;
        }

        System.out.println("Hf-Preprocessor Results: ");
        System.out.println(serviceHf.getIntPredictFromArrowTensorResult(arrowResults));

        Map<String, ArrowTensor> npsResults = new HashMap<>();
        try {
            npsResults = serviceNps.predictBlocking(modelName, serviceResults, Collections.emptyMap());
        } catch (IOException e) {
            System.out.println("NPS Exception: " + e.getMessage());
            return;
        }

        System.out.println("NPS Results:");
        System.out.println(serviceNps.getFloatVectorsFromNpsResult(npsResults, "sentence_embedding"));
    }
}
