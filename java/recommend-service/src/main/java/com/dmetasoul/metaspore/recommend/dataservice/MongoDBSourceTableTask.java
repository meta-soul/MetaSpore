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
import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.datasource.DataSource;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;

@SuppressWarnings("rawtypes")
@Slf4j
@DataServiceAnnotation("SourceTable")
public class SourceTableTask extends DataService {

    private DataSource dataSource;
    private FeatureConfig.Source source;

    @Override
    public boolean initService() {
        log.info("test refresh name:{} initService+++++++++++++++++++++++++++++++++++++++++", name);
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(name);
        dataSource = taskServiceRegister.getDataSources().get(sourceTable.getSource());
        source = taskFlowConfig.getSources().get(sourceTable.getSource());
        return true;
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        ServiceRequest req = super.makeRequest(depend, request, context);
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(name);
        if (Objects.equals(source.getKind(), "redis")) {  // redis source need keys param
            if (CollectionUtils.isNotEmpty(req.getKeys())) {
                return req;
            }
            List<String> keys = Lists.newArrayList();
            String key = sourceTable.getColumnNames().get(0);
            if (MapUtils.isNotEmpty(req.getEqConditions())) {
                Object value = req.getEqConditions().get(key);
                if (value != null) {
                    if (StringUtils.isEmpty(sourceTable.getPrefix())) {
                        keys.add(String.valueOf(value));
                    } else {
                        keys.add(String.format("%s_%s", sourceTable.getPrefix(), String.valueOf(value)));
                    }
                }
            }
            if (MapUtils.isNotEmpty(req.getInConditions())) {
                List<Object> value = req.getInConditions().get(key);
                if (CollectionUtils.isNotEmpty(value)) {
                    if (StringUtils.isEmpty(sourceTable.getPrefix())) {
                        keys.addAll(value.stream().map(String::valueOf).collect(Collectors.toList()));
                    } else {
                        keys.addAll(value.stream().map(x -> String.format("%s_%s", sourceTable.getPrefix(), String.valueOf(x))).collect(Collectors.toList()));
                    }
                }
            }
            if (keys.isEmpty()) {
                Object value = req.get(key);
                if (value != null) {
                    if (value instanceof Collection) {
                        List<Object> data = Lists.newArrayList();
                        data.addAll((Collection<?>) value);
                        if (StringUtils.isEmpty(sourceTable.getPrefix())) {
                            keys.addAll(data.stream().map(String::valueOf).collect(Collectors.toList()));
                        } else {
                            keys.addAll(data.stream().map(x -> String.format("%s_%s", sourceTable.getPrefix(), String.valueOf(x))).collect(Collectors.toList()));
                        }
                    }else {
                        if (StringUtils.isEmpty(sourceTable.getPrefix())) {
                            keys.add(String.valueOf(value));
                        } else {
                            keys.add(String.format("%s_%s", sourceTable.getPrefix(), String.valueOf(value)));
                        }
                    }
                }
            }
            if (keys.isEmpty()) {
                log.error("redis request loss keys makeRequest fail!");
                throw new RuntimeException("redis request loss keys makeRequest fail!");
            }
            req.setKeys(keys);
        } else if (source.getKind().equals("jdbc") || source.getKind().equals("mongodb")) {
            if (source.getKind().equals("jdbc") && StringUtils.isNotEmpty(req.getJdbcSql())) {
                return req;
            }
            if (req.getEqConditions() == null) req.setEqConditions( Maps.newHashMap());
            Map<String, Object> eqData = req.getEqConditions();
            if (MapUtils.isNotEmpty(eqData)) {
                eqData.keySet().forEach(x->{
                    if (!sourceTable.getColumnMap().containsKey(x)) {
                        eqData.remove(x);
                    }
                });
            }
            if (req.getInConditions() == null) req.setInConditions(Maps.newHashMap());
            Map<String, List<Object>> inData = req.getInConditions();
            if (MapUtils.isNotEmpty(inData)) {
                inData.keySet().forEach(x->{
                    if (!sourceTable.getColumnMap().containsKey(x)) {
                        inData.remove(x);
                    }
                });
            }
            if (MapUtils.isEmpty(eqData) && MapUtils.isEmpty(inData)) {
                if (MapUtils.isNotEmpty(req.getData())) {
                    req.getData().forEach((k, v) -> {
                        if (sourceTable.getColumnMap().containsKey(k)) {
                            if (v instanceof Collection) {
                                List<Object> data = Lists.newArrayList();
                                data.addAll((Collection<?>) v);
                                inData.put(k, data);
                            } else {
                                eqData.put(k, v);
                            }
                        }
                    });
                }
                if (eqData.isEmpty() && inData.isEmpty()) {
                    log.error("jdbc or mongodb request loss condition makeRequest fail!");
                    throw new RuntimeException("jdbc or mongodb request loss condition makeRequest fail!");
                }
            }

        }
        return req;
    }

    @Override
    public boolean checkResult(DataResult result) {
        if (result == null) {
            log.warn("result is null!");
            return false;
        }
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(name);
        for (String col : sourceTable.getColumnNames()) {
            String type = sourceTable.getColumnMap().get(col);
            Class dataClass = DataTypes.getDataClass(type);
            if (MapUtils.isNotEmpty(result.getValues())) {
                Map<String, Object> data = result.getValues();
                Object value = data.get(col);
                if (value != null && !dataClass.isInstance(value)) {
                    log.warn("sourceTable {} get result col:{} type is wrong, value:{}", name, col, value);
                    // return false;
                }
            }
            if (CollectionUtils.isNotEmpty(result.getData())) {
                for (Map data : result.getData()) {
                    Object value = data.get(col);
                    if (value != null && !dataClass.isInstance(value)) {
                        log.warn("sourceTable {} get result col:{} type is wrong, value:{}", name, col, value);
                        // return false;
                    }
                }
            }
        }
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult dataResult;
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(name);
        int retryNum = 0;
        long timeOut = 3000L;
        TimeUnit timeOutUnit = TimeUnit.MILLISECONDS;
        Map<String, Object> options = sourceTable.getOptions();
        if (MapUtils.isNotEmpty(options)) {
            retryNum = (int) options.getOrDefault("retryNum", 0);
            timeOut = (long) options.getOrDefault("timeOut", 30000L);
        }
        retryNum += 1;
        do {
            CompletableFuture<DataResult> future = dataSource.execute(makeRequest(sourceTable.getSource(), request, context), context);
            try {
                dataResult = future.get(timeOut, timeOutUnit);
                if (context.getStatus(source.getName(), name) == TaskStatusEnum.SUCCESS) {
                    return dataResult;
                }
                retryNum -= 1;
            } catch (InterruptedException | ExecutionException e) {
                log.error("there was an error when executing the CompletableFuture",e);
            } catch (TimeoutException e) {
                log.error("when task timeout!",e);
            }
        } while (retryNum >= 0);
        return null;
    }
}
