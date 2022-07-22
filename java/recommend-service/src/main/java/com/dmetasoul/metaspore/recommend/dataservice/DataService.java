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

import com.dmetasoul.metaspore.recommend.TaskServiceRegister;
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.ResultTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.*;

@Slf4j
public abstract class DataService {
    protected String name;
    protected ExecutorService workFlowPool;
    protected TaskFlowConfig taskFlowConfig;
    protected List<Chain> chains = Lists.newArrayList();
    protected Set<String> processedTask = Sets.newHashSet();
    protected Map<String, DataService> dataServices;
    protected List<String> lastChainTasks;
    protected boolean lastAny = false;
    protected TaskServiceRegister taskServiceRegister;

    public boolean init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister taskServiceRegister, ExecutorService workFlowPool) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null or taskServices is null , init fail!");
            return false;
        }
        this.name = name;
        this.workFlowPool = workFlowPool;
        this.dataServices = taskServiceRegister.getDataServices();
        this.taskFlowConfig = taskFlowConfig;
        this.taskServiceRegister = taskServiceRegister;
        return initService();
    }
    protected abstract boolean initService();
    public void close() {}

    public boolean checkResult(DataResult result) {
        if (result == null) {
            log.warn("result is null!");
            return false;
        }
        if (!result.isVaild()) {
            log.warn("result is error!");
            return false;
        }
        if ((result.getResultType() == ResultTypeEnum.VALUES && result.getValues() == null) ||
                (result.getResultType() == ResultTypeEnum.DATA && result.getData() == null) ||
                (result.getResultType() == ResultTypeEnum.FEATUREARRAYS && result.getFeatureArray() == null) ||
                (result.getResultType() == ResultTypeEnum.FEATURETABLE && result.getFeatureTable() == null)) {
            log.warn("result data is null!");
            return false;
        }
        result.setName(name);
        return true;
    }

    public DataResult getDataResultByName(String name, DataContext context) {
        DataResult result = context.getResult(name);
        if (!checkResult(result)) {
            log.error("name ：{} result get wrong！", name);
            return null;
        }
        return result;
    }

    public List<DataResult> getDataResultByNames(List<String> names, DataContext context) {
        List<DataResult> dataResults = Lists.newArrayList();
        for (String name : names) {
            DataResult result = getDataResultByName(name, context);
            if (result != null) {
                dataResults.add(result);
            }
        }
        return dataResults;
    }
    public DataResult getDataResultByRely(String name, DataContext context) {
        DataResult result = getDataResultByName(name, context);
        if (result == null) {
            log.error("rely：{} result get wrong！", name);
            throw new RuntimeException(String.format("task: %s get result fail!", name));
        }
        return result;
    }

    public List<DataResult> getDataResultByRelys(List<String> names, boolean isAny, DataContext context) {
        List<DataResult> dataResults = getDataResultByNames(names, context);
        if (isAny && dataResults.isEmpty()) {
            throw new RuntimeException(String.format("task: %s get rely result fail any at empty !", name));
        }
        if (!isAny && dataResults.size() != names.size()) {
            throw new RuntimeException(String.format("task: %s get rely result fail not any but loss!", name));
        }
        return dataResults;
    }

    public List<Map> getDataByColumns(DataResult dataResult, List<String> columnNames) {
        List<Map> data = Lists.newArrayList();
        if (MapUtils.isNotEmpty(dataResult.getValues())) {
            Map<String, Object> map = Maps.newHashMap();
            for (String col : columnNames) {
                Object value = dataResult.getValues().get(col);
                map.put(col, value);
            }
            data.add(map);
        } else if (CollectionUtils.isNotEmpty(dataResult.getData())) {
            for (Map item : dataResult.getData()) {
                Map<String, Object> map = Maps.newHashMap();
                for (String col : columnNames) {
                    Object value = item.get(col);
                    map.put(col, value);
                }
                data.add(map);
            }
        } else if (dataResult.getFeatureArray() != null) {
            DataResult.FeatureArray featureArray = dataResult.getFeatureArray();
            for (int index = 0; index < featureArray.getMaxIndex(); ++index) {
                Map<String, Object> map = Maps.newHashMap();
                for (String col : columnNames) {
                    Object value = featureArray.get(col, index);
                    map.put(col, value);
                }
                data.add(map);
            }
        }
        return data;
    }

    public List<Map> getTaskResultByColumns(List<String> taskNames, boolean isAny, List<String> columnNames, DataContext context) {
        List<DataResult> dataResults = getDataResultByRelys(taskNames, isAny, context);
        List<Map> data = Lists.newArrayList();
        for (DataResult dataResult : dataResults) {
            data.addAll(getDataByColumns(dataResult, columnNames));
        }
        return data;
    }

    public ServiceRequest makeRequest(String name, ServiceRequest request, DataContext context) {
        return new ServiceRequest(request, context);
    }

    public ServiceRequest makeRequest(DataContext context) {
        return new ServiceRequest(context);
    }

    public DataResult execute(DataContext context) {
        return execute(name, makeRequest(context), context);
    }

    public DataResult execute(String taskName, DataContext context) {
        return execute(taskName, makeRequest(context), context);
    }
    public DataResult execute(String taskName, ServiceRequest request, DataContext context) {
        DataService dataService = dataServices.get(taskName);
        if (dataService == null) {
            log.error("task:{} depend:{} service init fail!", name, taskName);
            return null;
        }
        // 调用服务为被调用任务构建请求数据
        ServiceRequest taskRequest = makeRequest(taskName, request, context);
        if (taskRequest == null) {
            log.error("task:{} request init fail!", name);
            return null;
        }
        dataService.execute(taskRequest, context);
        processedTask.add(taskName);
        return getDataResultByName(taskName, context);
    }

    public List<DataResult> execute(List<String> names, boolean isWhen, boolean isAny, ServiceRequest request, DataContext context) {
        Chain chain = new Chain();
        if (isWhen) {
            chain.setWhen(names);
            chain.setAny(isAny);
        } else {
            chain.setThen(names);
        }
        List<String> outputs = executeChain(chain, request, context);
        return getDataResultByNames(outputs, context);
    }

    public List<DataResult> execute(List<String> names, boolean isWhen, boolean isAny, DataContext context) {
        return execute(names, isWhen, isAny, null, context);
    }

    public List<DataResult> execute(List<String> names, DataContext context) {
        return execute(names, false, false, context);
    }

    public List<String> executeChain(Chain chain, DataContext context) {
        return executeChain(chain, makeRequest(context), context);
    }

    public List<String> executeChain(Chain chain, ServiceRequest request, DataContext context) {
        List<String> result = Lists.newArrayList();
        String lastThenTask = null;
        if (CollectionUtils.isNotEmpty(chain.getThen())) {
            for (String taskName : chain.getThen()) {
                lastThenTask = taskName;
                if (execute(taskName, request, context) == null) {
                    log.error("task:{} depend:{} exec fail!", name, taskName);
                    throw new RuntimeException(String.format("task:%s in %s exec fail!", taskName, name));
                }
            }
        }
        if (CollectionUtils.isNotEmpty(chain.getWhen())) {
            List<CompletableFuture<?>> whenList = Lists.newArrayList();
            for (String taskName : chain.getWhen()) {
                whenList.add(CompletableFuture.supplyAsync(() -> execute(taskName, request, context), workFlowPool)
                        .whenComplete(((dataResult, throwable) -> {
                            if (!checkResult(dataResult)) {
                                log.error("task:{} depend:{} exec fail!", name, taskName);
                            }
                            if (throwable != null) {
                                log.error("exception:{}", throwable.getMessage());
                            }
                        }))
                );
            }
            CompletableFuture<?> resultFuture;
            if (chain.isAny()) {
                resultFuture = CompletableFuture.anyOf(whenList.toArray(new CompletableFuture[]{}));
            } else {
                resultFuture = CompletableFuture.allOf(whenList.toArray(new CompletableFuture[]{}));
            }
            try {
                resultFuture.get(chain.getTimeOut(), chain.getTimeOutUnit());
            } catch (InterruptedException | ExecutionException e) {
                log.error("there was an error when executing the CompletableFuture", e);
            } catch (TimeoutException e) {
                log.error("when task timeout!", e);
            }
            result.addAll(chain.getWhen());
        } else if (StringUtils.isNotEmpty(lastThenTask)) {
            result.add(lastThenTask);
        }
        lastChainTasks = result;
        lastAny = chain.isAny();
        return result;
    }

    public DataResult execute(ServiceRequest request, DataContext context){
        // 1, 跟上次请求没变化，则直接使用上次处理结果
        DataResult result = getDataResultByName(name, context);
        String reqSign = request.genRequestSign();
        if (result != null && result.getReqSign().equals(reqSign)) {
            return result;
        }
        // 2, 执行chain，计算依赖服务结果
        for (Chain chain : chains) {
            List<String> taskNames = executeChain(chain, request, context);
            if (StringUtils.isNotEmpty(chain.getName())) {
                if (CollectionUtils.isEmpty(taskNames)) {
                    log.error("task:{} chain:{} no task!", name, chain.getName());
                    throw new RuntimeException(String.format("task:%s chain:%s no task!", name, chain.getName()));
                }
                DataResult chainResult = new DataResult();
                List<Map> data = getTaskResultByColumns(taskNames, chain.isAny(), chain.getColumnNames(), context);
                chainResult.setData(data);
                context.setResult(chain.getName(), chainResult);
            }
        }
        // 3, 执行服务处理函数
        result = process(request, context);
        if (checkResult(result)) {
            result.setReqSign(reqSign);
            context.setResult(name, result);
        } else {
            throw new RuntimeException(String.format("task:%s exec fail!", name));
        }
        return result;
    }

    protected abstract DataResult process(ServiceRequest request, DataContext context);
}
