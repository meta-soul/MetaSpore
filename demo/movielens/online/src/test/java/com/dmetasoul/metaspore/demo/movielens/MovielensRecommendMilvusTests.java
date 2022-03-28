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

package com.dmetasoul.metaspore.demo.movielens;

import com.dmetasoul.metaspore.demo.movielens.service.MilvusService;
import io.milvus.response.SearchResultsWrapper;
import org.assertj.core.util.Lists;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;

import java.util.List;
import java.util.Map;
import java.util.Random;

@SpringBootTest
public class MovielensRecommendMilvusTests {

    @Autowired
    private ApplicationContext context;

    @Test
    void contextLoads() {
    }

    @Test
    public void testQueryMilvusByEmbedding() {
        System.out.println("Test query Milvus Service by embeddings:");
        MilvusService service = context.getBean(MilvusService.class);
        List<List<Float>> vectors = generateFloatVectors(3, 32);
        Map<Integer, List<SearchResultsWrapper.IDScore>> result = service.findByEmbeddingVectors(vectors, 5);
        System.out.println(result);
    }

    private List<List<Float>> generateFloatVectors(int count, int dimension) {
        Random ran = new Random(1234567890);
        List<List<Float>> vectors = Lists.newArrayList();
        for (int n = 0; n < count; ++n) {
            List<Float> vector = Lists.newArrayList();
            for (int i = 0; i < dimension; ++i) {
                vector.add(ran.nextFloat());
            }
            vectors.add(vector);
        }
        return vectors;
    }
}