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

import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.datasource.RedisSource;
import com.dmetasoul.metaspore.recommend.enums.RedisTypeEnum;
import com.google.common.collect.Lists;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.Collection;
import java.util.List;
import java.util.Map;
/**
 * 针对source.kind=redis的SourceTable的DataService的实现类
 * 调用redis DataSource获取redis中的数据
 * 注解DataServiceAnnotation 必须设置， value应设置为RedisSourceTable。
 * Created by @author qinyy907 in 14:24 22/08/01.
 */

@Slf4j
@ServiceAnnotation("RedisSourceTable")
public class RedisSourceTableTask extends SourceTableTask {

    private RedisSource dataSource;
    private String columnKey;
    private String keyFormat;
    private RedisTypeEnum redisType;

    @Override
    public boolean initService() {
        if (super.initService() && source.getKind().equalsIgnoreCase("redis")) {
            dataSource = (RedisSource) taskServiceRegister.getDataSources().get(sourceTable.getSource());
        }
        if (StringUtils.isNotEmpty(sourceTable.getPrefix())) {
            keyFormat = String.format("%s_%%s", sourceTable.getPrefix());
        }
        if (sourceTable.getColumnNames().size() < 2) {
            return false;
        }
        List<String> columnNames = sourceTable.getColumnNames();
        columnKey = columnNames.get(0);
        redisType = RedisTypeEnum.getEnumByName(getOptionOrDefault("redisType", "hash"));
        redisType.init(keyFormat, dataSource.getRedisTemplate(), columnNames);
        return true;
    }

    @SuppressWarnings("rawtypes")
    private void fillDataList(Object value, List<Map<String, Object>> list, int limit) {
        if (value instanceof Collection) {
            for (Object item : (Collection)value) {
                list.addAll(redisType.process(String.valueOf(item), limit));
            }
        } else {
            list.addAll(redisType.process(String.valueOf(value),  limit));
        }
    }

    @Override
    protected List<Map<String, Object>> processRequest(ServiceRequest request, DataContext context) {
        Map<String, Object> data = request.getData();
        int limit = request.getLimit();
        List<Map<String, Object>> list = Lists.newArrayList();
        if (MapUtils.isNotEmpty(data) && data.containsKey(columnKey)) {
            Object value = data.get(columnKey);
            fillDataList(value, list, limit);
        }
        return list;
    }
}
