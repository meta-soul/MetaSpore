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
import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.zaxxer.hikari.HikariDataSource;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.actuate.health.Status;
import org.springframework.dao.support.DataAccessUtils;
import org.springframework.jdbc.IncorrectResultSetColumnCountException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.support.JdbcUtils;
import org.springframework.util.StringUtils;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.List;
import java.util.Map;

/**
 * source.kind=jdbc的DataSource实现类
 * 配置中的source.kind需要与注解DataSourceAnnotation中value保持一致
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@Slf4j
@Data
@ServiceAnnotation("JDBC")
public class JDBCSource extends DataSource {
    private JdbcTemplate jdbcTemplate;
    private NamedParameterJdbcTemplate namedTemplate;
    private HikariDataSource dataSource;
    private String validationQuery;

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
        validationQuery = CommonUtils.getField(source.getOptions(), "checkSql");
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
    @Override
    public void doHealthCheck(Status status, Map<String, Object> details, Throwable exception) throws Exception {
        super.doHealthCheck(status, details, exception);
        if (dataSource == null) {
            status = Status.DOWN;
            details.put("database", "unknown");
            return;
        }
        details.put("database", this.jdbcTemplate.execute(this::getProduct));
        if (StringUtils.hasText(validationQuery)) {
            details.put("validationQuery", validationQuery);
            List<Object> results = this.jdbcTemplate.query(validationQuery, new SingleColumnRowMapper());
            Object result = DataAccessUtils.requiredSingleResult(results);
            details.put("result", result);
        } else {
            details.put("validationQuery", "isValid()");
            boolean valid = Boolean.TRUE.equals(this.jdbcTemplate.execute(this::isConnectionValid));
            status = (valid ? Status.UP : Status.DOWN);
        }
    }

    private String getProduct(Connection connection) throws SQLException {
        return connection.getMetaData().getDatabaseProductName();
    }

    private Boolean isConnectionValid(Connection connection) throws SQLException {
        return connection.isValid(0);
    }
    private static class SingleColumnRowMapper implements RowMapper<Object> {
        private SingleColumnRowMapper() {
        }

        public Object mapRow(ResultSet rs, int rowNum) throws SQLException {
            ResultSetMetaData metaData = rs.getMetaData();
            int columns = metaData.getColumnCount();
            if (columns != 1) {
                throw new IncorrectResultSetColumnCountException(1, columns);
            } else {
                return JdbcUtils.getResultSetValue(rs, 1);
            }
        }
    }
}
