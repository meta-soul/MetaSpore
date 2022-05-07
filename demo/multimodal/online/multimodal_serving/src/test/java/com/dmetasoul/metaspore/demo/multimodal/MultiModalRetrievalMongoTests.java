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

import com.dmetasoul.metaspore.demo.multimodal.domain.BaikeQaDemo;
import com.dmetasoul.metaspore.demo.multimodal.repository.BaikeQaDemoRepository;
import com.dmetasoul.metaspore.demo.multimodal.domain.TxtToImgDemo;
import com.dmetasoul.metaspore.demo.multimodal.repository.BaikeQaDemoRepository;
import com.dmetasoul.metaspore.demo.multimodal.repository.TxtToImgDemoRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import java.util.*;

@SpringBootTest
class MultiModalRetrievalMongoTests {
    @Autowired
    private BaikeQaDemoRepository baikeQaRepository;

    @Autowired
    private TxtToImgDemoRepository txtToImgDemoRepository;

    @Test
    void contextLoads() {
    }

    @Test
    public void testQueryMongoByIdOfBaikeQa() {
        List<String> ids = new ArrayList<String>(Arrays.asList("1", "2"));
        System.out.println("Test query Mongo(baike-qa) by id:");
        for (int i = 0; i < ids.size(); i++) {
            String id = ids.get(i);
            Optional<BaikeQaDemo> item = baikeQaRepository.findByQueryid(id);
            System.out.println(item);
        }
    }

    @Test
    public void testQueryMongoByIdsOfBaikeQa() {
        List<String> ids = new ArrayList<String>(Arrays.asList("1", "2"));
        System.out.println("Test query Mongo(baike-qa) by ids:");
        Collection<BaikeQaDemo> items = baikeQaRepository.findByQueryidIn(ids);
        System.out.println(items);
    }

    @Test
    public void testQueryMongoByIdOfTxtToImg() {
        List<String> ids = new ArrayList<String>(Arrays.asList("1", "2"));
        System.out.println("Test query Mongo(txt2img) by id:");
        for (int i = 0; i < ids.size(); i++) {
            String id = ids.get(i);
            Optional<TxtToImgDemo> item = txtToImgDemoRepository.findByQueryid(id);
            System.out.println(item);
        }
    }

    @Test
    public void testQueryMongoByIdsOfTxtToImg() {
        List<String> ids = new ArrayList<String>(Arrays.asList("1", "2"));
        System.out.println("Test query Mongo(txt2img) by ids:");
        Collection<TxtToImgDemo> items = txtToImgDemoRepository.findByQueryidIn(ids);
        System.out.println(items);
    }
}