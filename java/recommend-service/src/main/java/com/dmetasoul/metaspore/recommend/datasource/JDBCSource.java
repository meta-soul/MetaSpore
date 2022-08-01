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

import com.dmetasoul.metaspore.recommend.annotation.DataSourceAnnotation;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.zaxxer.hikari.HikariDataSource;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;

import java.util.*;
/**
 * source.kind=jdbc的DataSource实现类
 * 配置中的source.kind需要与注解DataSourceAnnotation中value保持一致
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataSourceAnnotation("jdbc")
public class JDBCSource extends DataSource {
    private JdbcTemplate jdbcTemplate;
    private NamedParameterJdbcTemplate namedTemplate;
    private HikariDataSource dataSource;

    @Override
    public boolean initService() {
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        String uri = (String) source.getOptions().get("uri");
        String driver = (String) source.getOptions().get("driver");
        String user = (String) source.getOptions().get("user");
        String password = (String) source.getOptions().get("password");
        int mixPoolSize = (int) source.getOptions().getOrDefault("maxPoolSize", 100);
        dataSource = new HikariDataSource();
        dataSource.setJdbcUrl(uri);
        dataSource.setDriverClassName(driver);
        dataSource.setMaximumPoolSize(mixPoolSize);
        dataSource.setUsername(user);
        dataSource.setPassword(password);
        jdbcTemplate = new JdbcTemplate(dataSource);
        namedTemplate = new NamedParameterJdbcTemplate(dataSource);
        return true;
    }

    @Override
    public void close() {
        if (dataSource != null) {
            try {
                dataSource.close();
            } catch (Exception ex) {
                log.error("jdbc dataSource close fail! {}", ex.getMessage());
            }
        }
    }
}
