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

import java.util.*;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataSourceAnnotation("jdbc")
public class JDBCSource extends DataSource {
    private JdbcTemplate jdbcTemplate;

    @Override
    public boolean initService() {
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        String uri = (String) source.getOptions().get("uri");
        String driver = (String) source.getOptions().get("driver");
        String user = (String) source.getOptions().get("user");
        String password = (String) source.getOptions().get("password");
        int mixPoolSize = (int) source.getOptions().getOrDefault("maxPoolSize", 100);
        HikariDataSource dataSource = new HikariDataSource();
        dataSource.setJdbcUrl(uri);
        dataSource.setDriverClassName(driver);
        dataSource.setMaximumPoolSize(mixPoolSize);
        dataSource.setUsername(user);
        dataSource.setPassword(password);
        jdbcTemplate = new JdbcTemplate(dataSource);
        return true;
    }

    @Override
    public boolean checkRequest(ServiceRequest request, DataContext context) {
        String sql = request.getJdbcSql();
        if (StringUtils.isNotEmpty(sql)) {
            return true;
        }
        if (MapUtils.isEmpty(request.getEqConditions()) && MapUtils.isEmpty(request.getInConditions())) {
            log.error("jdbc create query sql need condition!");
            return false;
        }
        return true;
    }

    private String getConditions(ServiceRequest request, List<String> columns) {
        Set<String> columnSet = Sets.newHashSet();
        columnSet.addAll(columns);
        List<String> parts = Lists.newArrayList();
        if (MapUtils.isNotEmpty(request.getEqConditions())) {
            request.getEqConditions().forEach((field, queryId) -> {
                if (!columnSet.contains(field)) {
                    throw new RuntimeException("jdbc request Condition field is not in columns!");
                }
                if (queryId instanceof String) {
                    parts.add(String.format("%s = \"%s\"", field, queryId));
                } else {
                    parts.add(String.format("%s = %s", field, queryId));
                }
            });
        }
        if (MapUtils.isNotEmpty(request.getInConditions())) {
            request.getInConditions().forEach((field, queryId) -> {
                if (!columnSet.contains(field)) {
                    throw new RuntimeException("jdbc request Condition field is not in columns!");
                }
                if (CollectionUtils.isEmpty(queryId)) {
                    throw new RuntimeException("jdbc request inCondition queryIds is empty!");
                }
                Object id = queryId.get(0);
                List<String> idList = Lists.newArrayList();
                if (id instanceof String) {
                    queryId.forEach(x-> idList.add(String.format("\"%s\"", x)));
                } else {
                    queryId.forEach(x-> idList.add(String.format("%s", x)));
                }
                parts.add(String.format("%s in (%s)", field, String.join(",", idList)));
            });
        }
        String sql = String.join(" and ", parts);
        if (CollectionUtils.isNotEmpty(request.getSqlFilters())) {
            if (StringUtils.isNotEmpty(sql)) {
                sql += " and ";
            }
            sql += String.join(" and ", request.getSqlFilters());
        }
        return sql;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = new DataResult();
        String parent = request.getParent();
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(parent);
        String sql = request.getJdbcSql();
        List<String> columns = sourceTable.getColumnNames();
        if (StringUtils.isEmpty(sql)) {
            String select = String.join(",", columns);
            String filter = getConditions(request, columns);
            if (CollectionUtils.isNotEmpty(sourceTable.getSqlFilters())) {
                if (StringUtils.isNotEmpty(filter)) {
                    filter += " and ";
                }
                filter += String.join(" and ", sourceTable.getSqlFilters());
            }
            int limit = request.getLimit();
            if (limit > 0) {
                filter += String.format(" limit %d", limit);
            }
            sql = String.format("select %s from %s where %s", select, sourceTable.getTable(), filter);
        }
        result.setData(jdbcTemplate.query(sql, rs -> {
            List<Map> list = Lists.newArrayList();
            while (rs.next()) {
                Map<String, Object> item = Maps.newHashMap();
                for (String col : columns) {
                    item.put(col, rs.getObject(col));
                }
                list.add(item);
            }
            return list;
        }));
        return result;
    }
}
