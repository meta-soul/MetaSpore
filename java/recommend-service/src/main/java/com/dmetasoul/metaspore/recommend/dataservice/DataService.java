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
    protected LinkedBlockingQueue<Chain> taskFlow;
    protected int executeNum = 1;

    protected Chain depend;
    protected Map<String, DataService> dataServices;
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
        this.taskFlow = new LinkedBlockingQueue<>();
        return initService();
    }
    protected abstract boolean initService();
    public void close() {}

    public boolean checkResult(DataResult result) {
        if (result == null) {
            return false;
        }
        if (!result.isVaild()) {
            log.warn("task: {} result is error!", name);
            return false;
        }
        if (result.isNull()) {
            log.warn("task: {} result data is null!", name);
            return false;
        }
        result.setName(name);
        return true;
    }

    public DataResult getDataResultByName(String taskName, DataContext context) {
        DataResult result = context.getResult(name, taskName);
        if (!checkResult(result)) {
            return null;
        }
        return result;
    }

    public List<DataResult> getDataResultByNames(List<String> names, DataContext context) {
        List<DataResult> dataResults = Lists.newArrayList();
        for (String taskName : names) {
            DataResult result = getDataResultByName(taskName, context);
            if (result != null) {
                dataResults.add(result);
            }
        }
        return dataResults;
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
        return dataResult.getData(columnNames);
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

    public Chain executeChain(Chain chain, ServiceRequest request, DataContext context) {
        Chain lastChain = new Chain();
        List<String> then = chain.getThen();
        if (CollectionUtils.isNotEmpty(then)) {
            int i = 0;
            for (; i < then.size(); ++i) {
                String taskName = then.get(i);
                if (execute(taskName, request, context) == null) {
                    log.warn("task:{} depend:{} exec fail!", name, taskName);
                    break;
                }
            }
            lastChain.setThen(then.subList(i, then.size()));
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
            } catch (InterruptedException | ExecutionException | TimeoutException e) {
                log.error("there was an error when executing the CompletableFuture", e);
            }
            List<String> noExecuteTasks = Lists.newArrayList();
            for (String taskName : chain.getWhen()) {
                if (getDataResultByName(taskName, context) == null) {
                    noExecuteTasks.add(taskName);
                }
            }
            if (chain.isAny() && noExecuteTasks.size() == chain.getWhen().size() || !chain.isAny() && !noExecuteTasks.isEmpty()) {
                lastChain.setWhen(noExecuteTasks);
                lastChain.setAny(chain.isAny());
            }
        }
        return lastChain;
    }

    public DataResult execute(String taskName, ServiceRequest request, DataContext context) {
        DataResult result = getDataResultByName(taskName, context);
        if (result != null) {
            return result;
        }
        DataService dataService = dataServices.get(taskName);
        if (dataService == null) {
            log.error("task:{} depend:{} service init fail!", name, taskName);
            return null;
        }
        // 调用服务为被调用任务构建请求数据
        ServiceRequest taskRequest = makeRequest(taskName, request, context);
        if (taskRequest == null) {
            return null;
        }
        result = dataService.execute(taskRequest, context);
        if (checkResult(result)) {
            // 同一个任务下的依赖子任务不重复计算，子任务请求一般不会变化
            context.setResult(name, taskName, result);
            return result;
        }
        return null;
    }

    protected void preCondition(ServiceRequest request, DataContext context) {
        if (depend != null && !depend.isEmpty()) {
            taskFlow.offer(depend);
        }
    }


    public DataResult execute(ServiceRequest request, DataContext context){
        // 0, 跟上次请求没变化，则直接使用上次处理结果
        DataResult result = getDataResultByName(name, context);
        String reqSign = request.genRequestSign();
        if (result != null && result.getReqSign().equals(reqSign)) {
            return result;
        }
        // 1, 执行depend任务前预处理
        preCondition(request, context);
        // 2, 执行chain，计算依赖depend服务结果
        int num = executeNum * taskFlow.size();
        Chain chain = taskFlow.poll();
        while (chain != null && !chain.isEmpty()) {
            Chain newChain = executeChain(chain, request, context);
            if (newChain != null && !newChain.isEmpty()) {
                taskFlow.offer(newChain);
                if (newChain.noChanged(chain)) {
                    num -= 1;
                    if (num < 0) {
                        break;
                    }
                }
            }
            chain = taskFlow.poll();
        }
        if (chain == null || chain.isEmpty()) {
            // 3, 执行服务处理函数
            result = process(request, context);
            if (checkResult(result)) {
                result.setReqSign(reqSign);
                // 缓存结果， 相同的请求不重复计算
                context.setResult(name, result);
                return result;
            }
        }
        throw new RuntimeException(String.format("task:%s exec fail!", name));
    }


    protected abstract DataResult process(ServiceRequest request, DataContext context);
}
