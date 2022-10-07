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

import com.dmetasoul.metaspore.recommend.baseservice.TaskServiceRegister;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.StopWatch;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;

/**
 * DataService的base类
 * DataService用于实现各种数据获取和计算的任务，DataService Task互相组织在一起构成一系列任务流，来完成实现推荐服务中的服务功能
 * DataService 依赖配置中的FeatureConfig， 服务实例注册在类TaskServiceRegister中。
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@Slf4j
public abstract class DataService {
    /**
     * DataService 名称 与配置feature-config中sourcetable， feature， algoInference， algotransform的name对应
     */
    protected String name;
    /**
     * 用于执行DataService任务流的线程池
     */
    protected ExecutorService workFlowPool;
    /**
     * 配置数据对象
     */
    protected TaskFlowConfig taskFlowConfig;
    /**
     * DataService的依赖任务队列。
     * 为了支持更灵活的DataService任务，相关依赖任务执行过程中，可以根据需要动态的添加相关的依赖任务，来完成更复杂的数据转换计算任务
     */
    protected LinkedBlockingQueue<Chain> taskFlow;
    /**
     * DataService的依赖任务流chain未能执行成功重新加入执行队列的次数。
     */
    protected int executeNum = 1;
    /**
     * 对于某些依赖任务执行比较确定的DataService，初始化过程中确定的依赖任务执行流程chain。
     */
    protected Chain depend;

    protected List<Field> resFields;
    protected List<DataTypeEnum> dataTypes;
    /**
     * DataService实例注册类对象
     * 服务实例注册在类TaskServiceRegister中。
     */
    protected TaskServiceRegister taskServiceRegister;

    /**
     * DataService base 类初始化， 外部使用DataService需要调用此函数进行初始化
     * DataService实例不能通过Autowired注解来获取配置数据对象，DataService实例注册类对象和线程池对象，所以需要在init初始化的时候加载进来
     */
    public boolean init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister taskServiceRegister, ExecutorService workFlowPool) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null or taskServices is null , init fail!");
            return false;
        }
        this.name = name;
        this.workFlowPool = workFlowPool;
        this.taskFlowConfig = taskFlowConfig;
        this.taskServiceRegister = taskServiceRegister;
        this.taskFlow = new LinkedBlockingQueue<>();
        resFields = Lists.newArrayList();
        dataTypes = Lists.newArrayList();
        return initService();
    }

    /**
     * DataService 具体实现子类初始化， 由base类init函数调用，对外不可见
     */
    protected abstract boolean initService();

    /**
     * 用于DataService 具体实现子类关闭外部服务连接等操作
     */
    public void close() {
    }

    /**
     * 用于验证DataService生成的result是否正确，保证输出的result为正确结果
     */
    public boolean checkResult(DataResult result) {
        if (result == null) {
            return false;
        }
        if (result.isNull()) {
            log.warn("task: {} result data is null!", name);
            return false;
        }
        return true;
    }

    /**
     * 根据DataService的依赖任务的名称taskName，获取由该DataService调用taskName的任务生成的结果
     * 这里认为同一个DataService在执行同一个请求的过程中，多次执行依赖任务taskName，获取的都是相同的数据结果
     */
    public DataResult getDataResultByName(String taskName, DataContext context) {
        return getDataResultByName(name, taskName, context);
    }

    public DataResult getDataResultByName(String parent, String taskName, DataContext context) {
        if (StringUtils.isEmpty(parent)) parent = name;
        DataResult result;
        if (parent.equals(taskName)) {
            result = context.getResult(parent);
        } else {
            result = context.getResult(parent, taskName);
        }
        if (!checkResult(result)) {
            return null;
        }
        return result;
    }

    public void setDataResultByName(String taskName, DataResult result, DataContext context) {
        if (!checkResult(result)) {
            log.warn("set: {}.{} result data is null!", name, taskName);
            return;
        }
        context.setResult(name, taskName, result);
    }

    /**
     * 根据DataService的依赖任务的名称集合names，获取相关的任务生成的结果集合, 获取不到结果的直接忽略
     */
    public List<DataResult> getDataResultByNames(List<String> names, DataContext context) {
        List<DataResult> dataResults = Lists.newArrayList();
        if (CollectionUtils.isNotEmpty(names)) {
            for (String taskName : names) {
                DataResult result = getDataResultByName(taskName, context);
                if (result != null) {
                    dataResults.add(result);
                }
            }
        }
        return dataResults;
    }

    public DataResult setDataResult(List<Map<String, Object>> res) {
        if (res == null) {
            return null;
        }
        DataResult result = new DataResult();
        FeatureTable featureTable = new FeatureTable(name, resFields);
        result.setFeatureTable(featureTable);
        result.setDataTypes(dataTypes);
        if (CollectionUtils.isEmpty(res)) {
            featureTable.finish();
            return result;
        }
        for (int i = 0; i < resFields.size(); ++i) {
            DataTypeEnum dataType = dataTypes.get(i);
            Field field = resFields.get(i);
            String col = field.getName();
            for (int index = 0; index < res.size(); ++index) {
                Map<String, Object> map = res.get(index);
                if (!dataType.set(featureTable, col, index, map.get(col))) {
                    log.error("set featuraTable fail!");
                }
            }
        }
        featureTable.finish();
        return result;
    }

    public DataResult setDataResult(Map<String, List<Object>> res) {
        if (res == null) {
            return null;
        }
        DataResult result = new DataResult();
        FeatureTable featureTable = new FeatureTable(name, resFields);
        result.setFeatureTable(featureTable);
        result.setDataTypes(dataTypes);
        if (MapUtils.isEmpty(res)) {
            featureTable.finish();
            return result;
        }
        for (int i = 0; i < resFields.size(); ++i) {
            DataTypeEnum dataType = dataTypes.get(i);
            Field field = resFields.get(i);
            String col = field.getName();
            if (!dataType.set(featureTable, col, res.get(col))) {
                log.error("set featureTable fail!");
            }
        }
        featureTable.finish();
        return result;
    }

    /**
     * 从DataResult中提取相关字段集合的数据List<Map>
     */
    public List<List<Object>> getDataByColumns(DataResult dataResult, List<String> columnNames) {
        List<List<Object>> res = Lists.newArrayList();
        if (CollectionUtils.isNotEmpty(columnNames)) {
            for (String col : columnNames) {
                List<Object> values = dataResult.get(col);
                if (values == null) {
                    log.error("featuraTable not has col:{}!", col);
                    values = Lists.newArrayList();
                }
                res.add(values);
            }
        }
        return res;
    }

    /**
     * 基于当前DataService的请求数据和上下文数据，构建依赖任务name的请求
     */
    public ServiceRequest makeRequest(String name, ServiceRequest request, DataContext context) {
        return new ServiceRequest(request, context);
    }

    /**
     * 基于上下文数据，构建当前任务的请求
     */
    public ServiceRequest makeRequest(DataContext context) {
        return new ServiceRequest(context);
    }

    /**
     * 基于上下文数据，构建当前任务的请求，来执行当前的DataService
     */
    public DataResult execute(DataContext context) {
        return execute(name, makeRequest(context), context);
    }

    /**
     * 执行DataService所依赖的任务执行流chain
     */
    public Chain executeChain(Chain chain, ServiceRequest request, DataContext context) {
        StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
        try {
            timeRecorder.start(String.format("%s_executeChain", name));
            // lastChain 用于记录未执行成功的task
            Chain lastChain = new Chain();
            List<String> then = chain.getThen();
            // 顺序执行任务then
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
            // 并行执行任务when
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
                // 设置any or all
                if (chain.isAny()) {
                    resultFuture = CompletableFuture.anyOf(whenList.toArray(new CompletableFuture[]{}));
                } else {
                    resultFuture = CompletableFuture.allOf(whenList.toArray(new CompletableFuture[]{}));
                }
                // 获取并发执行结果
                try {
                    resultFuture.get(chain.getTimeOut(), chain.getTimeOutUnit());
                } catch (InterruptedException | ExecutionException | TimeoutException e) {
                    log.error(String.format("the service: %s there was an error when executing the CompletableFuture", name), e);
                }
                // 记录未执行成功的when任务
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
        } finally {
            timeRecorder.stop();
            context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
        }
    }

    /**
     * 正确执行DataService所依赖的任务taskName之后，所进行的操作
     */
    public void afterProcess(String taskName, ServiceRequest request, DataContext context) {
    }

    /**
     * 执行DataService所依赖的任务taskName
     */
    public DataResult execute(String taskName, ServiceRequest request, DataContext context) {
        StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
        try {
            timeRecorder.start(String.format("%s_execute", name));
            // 如果任务已经被执行过，则直接获取结果
            DataResult result = getDataResultByName(taskName, context);
            if (result != null) {
                return result;
            }
            if (StringUtils.isNotEmpty(request.getParent())) {
                result = getDataResultByName(request.getParent(), taskName, context);
                if (result != null) {
                    context.setResult(name, taskName, result);
                    return result;
                }
            }
            DataService dataService = taskServiceRegister.getDataService(taskName);
            if (dataService == null) {
                log.error("task:{} depend:{} service init fail!", name, taskName);
                return null;
            }
            // 调用服务为被调用任务构建请求数据
            ServiceRequest taskRequest = makeRequest(taskName, request, context);
            if (taskRequest == null) {
                return null;
            }
            taskRequest.setParent(name);
            result = dataService.execute(taskRequest, context);
            if (checkResult(result)) {
                context.setResult(name, taskName, result);
                // 根据需要，执行taskName执行完毕后的处理逻辑
                afterProcess(taskName, request, context);
                return result;
            }
            return null;
        } finally {
            timeRecorder.stop();
            context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
        }
    }

    /**
     * 执行DataService任务前，进行任务的预处理操作
     */
    protected void preCondition(ServiceRequest request, DataContext context) {
        if (depend != null && !depend.isEmpty()) {
            taskFlow.offer(depend);
        }
    }

    /**
     * 执行DataService任务流程
     */
    public DataResult execute(ServiceRequest request, DataContext context) {
        // 0, 跟上次请求没变化，则直接使用上次处理结果
        DataResult result = getDataResultByName(name, context);
        String reqSign = request.genRequestSign();
        if (result != null && result.getReqSign().equals(reqSign)) {
            return result;
        }
        StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
        timeRecorder.start(String.format("%s_execute_pre_depend", name));
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
        timeRecorder.stop();
        if (chain == null || chain.isEmpty()) {
            try {
                timeRecorder.start(String.format("%s_process", name));
                // 3, 执行服务处理函数
                result = process(request, context);
                if (checkResult(result)) {
                    result.setReqSign(reqSign);
                    result.setName(name);
                    // 缓存结果， 相同的请求不重复计算
                    context.setResult(name, result);
                    return result;
                }
            } finally {
                timeRecorder.stop();
                context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
            }
        }
        context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
        throw new RuntimeException(String.format("task:%s exec fail!", name));
    }

    /**
     * DataService任务具体的处理流程
     */
    protected abstract DataResult process(ServiceRequest request, DataContext context);
}
