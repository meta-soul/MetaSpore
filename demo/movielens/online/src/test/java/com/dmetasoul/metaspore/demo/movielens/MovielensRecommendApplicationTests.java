package com.dmetasoul.metaspore.demo.movielens;

import com.dmetasoul.metaspore.demo.movielens.domain.Item;
import com.dmetasoul.metaspore.demo.movielens.domain.MilvusItemId;
import com.dmetasoul.metaspore.demo.movielens.domain.Swing;
import com.dmetasoul.metaspore.demo.movielens.domain.User;
import com.dmetasoul.metaspore.demo.movielens.repository.ItemRepository;
import com.dmetasoul.metaspore.demo.movielens.repository.MilvusItemIdRepository;
import com.dmetasoul.metaspore.demo.movielens.repository.SwingRepository;
import com.dmetasoul.metaspore.demo.movielens.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.*;

@SpringBootTest
class MovielensRecommendApplicationTests {
    @Autowired
    private ItemRepository itemRepository;
    @Autowired
    private UserRepository userRepository;
    @Autowired
    private SwingRepository swingRepository;
    @Autowired
    private MilvusItemIdRepository milvusItemIdRepository;


    @Test
    void contextLoads() {
    }

    List<String> ids = new ArrayList<String>(Arrays.asList("10", "100"));

    @Test
    public void testInsertMysql() {
        for (int i = 0; i < ids.size(); i++) {
            String id = ids.get(i);
            itemRepository.save(Item.builder()
                    .queryid(id)
                    .title("title_" + id)
                    .build());
        }
    }


    @Test
    public void testQueryMongoById() {
        System.out.println("Test query Mongo by id:");

        for (int i = 0; i < ids.size(); i++) {
            String id = ids.get(i);
            // Optional<Item> item = itemRepository.findByQueryid(id);
            Optional<User> user = userRepository.findByQueryid(id);
            Optional<Swing> swing = swingRepository.findByQueryid(id);

            // System.out.println(item);
            System.out.println(user);
            System.out.println(swing);

        }
    }

    @Test
    public void testQueryMongoByIds() {
        System.out.println("Test query Mongo by ids:");

        Collection<Item> items = itemRepository.findByQueryidIn(ids);
        Collection<User> users = userRepository.findByQueryidIn(ids);
        Collection<MilvusItemId> milvusItemId = milvusItemIdRepository.findByQueryidIn(List.of("17179869312", "8589934767"));

        System.out.println(items);
        System.out.println(users);
        System.out.println(milvusItemId);
    }


}
