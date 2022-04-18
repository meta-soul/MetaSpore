package com.dmetasoul.metaspore.demo.multimodal;

import com.dmetasoul.metaspore.demo.multimodal.domain.BaikeQaDemo;
import com.dmetasoul.metaspore.demo.multimodal.repository.BaikeQaDemoRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import java.util.*;

@SpringBootTest
class MultiModalRetrievalMongoTests {
    @Autowired
    private BaikeQaDemoRepository baikeQaRepository;

    @Test
    void contextLoads() {
    }

    @Test
    public void testQueryMongoByIdOfBaikeQa() {
        List<String> ids = new ArrayList<String>(Arrays.asList("1", "2"));
        System.out.println("Test query Mongo by id:");
        for (int i = 0; i < ids.size(); i++) {
            String id = ids.get(i);
            Optional<BaikeQaDemo> item = baikeQaRepository.findByQueryid(id);
            System.out.println(item);
        }
    }

    @Test
    public void testQueryMongoByIdsOfBaikeQa() {
        List<String> ids = new ArrayList<String>(Arrays.asList("1", "2"));
        System.out.println("Test query Mongo by ids:");
        Collection<BaikeQaDemo> items = baikeQaRepository.findByQueryidIn(ids);
        System.out.println(items);
    }
}