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
