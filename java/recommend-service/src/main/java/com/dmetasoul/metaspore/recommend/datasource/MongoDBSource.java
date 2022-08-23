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
package com.dmetasoul.metaspore.recommend.datasource;

import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.SimpleMongoClientDatabaseFactory;
/**
 * source.kind=mongodb的DataSource实现类
 * 配置中的source.kind需要与注解DataSourceAnnotation中value保持一致
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@Slf4j
@Data
@ServiceAnnotation("MongoDB")
public class MongoDBSource extends DataSource {
    private MongoTemplate mongoTemplate;
    private SimpleMongoClientDatabaseFactory simpleMongoClientDatabaseFactory;

    @Override
    public boolean initService() {
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        if (!source.getKind().equalsIgnoreCase("mongodb")) {
            log.error("config mongodb fail! is not kind:{} eq mongodb!", source.getKind());
            return false;
        }
        String uri = (String) source.getOptions().get("uri");
        simpleMongoClientDatabaseFactory = new SimpleMongoClientDatabaseFactory(uri);
        mongoTemplate = new MongoTemplate(simpleMongoClientDatabaseFactory);
        return true;
    }

    @Override
    public void close() {
        if (simpleMongoClientDatabaseFactory != null) {
            try {
                simpleMongoClientDatabaseFactory.destroy();
            } catch (Exception ex) {
                log.error("mongodb simpleMongoClientDatabaseFactory destroy fail! {}", ex.getMessage());
            }
        }
    }
}
