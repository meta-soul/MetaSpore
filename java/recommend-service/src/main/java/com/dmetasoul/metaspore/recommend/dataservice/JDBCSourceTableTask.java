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
package com.dmetasoul.metaspore.recommend.dataservice;

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.datasource.JDBCSource;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.*;
/**
 * 针对source.kind=jdbc的SourceTable的DataService的实现类
 * 调用关系数据库 DataSource获取数据库中的数据
 * 注解DataServiceAnnotation 必须设置， value应设置为JDBCSourceTable。
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@SuppressWarnings("rawtypes")
@Slf4j
@DataServiceAnnotation("JDBCSourceTable")
public class JDBCSourceTableTask extends SourceTableTask {

    private JDBCSource dataSource;

    private String jdbcSql;

    private String filterSql;
    private Set<String> columns;
    private List<String> columnNames;

    @Override
    public boolean initService() {
        if (super.initService() && source.getKind().equals("jdbc")) {
            dataSource = (JDBCSource) taskServiceRegister.getDataSources().get(sourceTable.getSource());
        }
        columns = sourceTable.getColumnMap().keySet();
        columnNames = sourceTable.getColumnNames();
        String select = String.join(",", columnNames);
        filterSql = String.join(" and ", sourceTable.getSqlFilters());
        jdbcSql = String.format("select %s from %s where ", select, sourceTable.getTable());
        return true;
    }

    private void fillParts(String col, Object value, List<String> parts, Map<String, Object> params) {
        if (value instanceof Collection) {
            List<Object> list = Lists.newArrayList();
            list.addAll((Collection)value);
            parts.add(String.format("%s in(:%s)", col, col));
            params.put(col, list);
        } else {
            parts.add(String.format("%s =:%s", col, col));
            params.put(col, value);
        }
    }

    @Override
    protected DataResult processRequest(ServiceRequest request, DataContext context) {
        Map<String, Object> data = request.getData();
        Map<String, Object> params = Maps.newHashMap();
        List<String> parts = Lists.newArrayList();
        for (String col : columns) {
            if (MapUtils.isNotEmpty(data) && data.containsKey(col)) {
                Object value = data.get(col);
                fillParts(col, value, parts, params);
            }
        }
        if (StringUtils.isNotEmpty(filterSql)) {
            parts.add(filterSql);
        }
        String sql = String.format("%s %s", jdbcSql, String.join(" and ", parts));
        if (request.getLimit() > 0) {
            sql += String.format(" limit %d", request.getLimit());
        }
        DataResult result = new DataResult();
        result.setData(dataSource.getNamedTemplate().query(sql, params, rs -> {
            List<Map> list = Lists.newArrayList();
            while (rs.next()) {
                Map<String, Object> item = Maps.newHashMap();
                for (String col : columnNames) {
                    item.put(col, rs.getObject(col));
                }
                list.add(item);
            }
            return list;
        }));
        return result;
    }
}
