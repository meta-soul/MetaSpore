package com.dmetasoul.metaspore.recommend.dataservice;

import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.ResultTypeEnum;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.dmetasoul.metaspore.recommend.TaskFlow;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.*;

@Slf4j
@Data
public abstract class DataService {
    protected String name;
    protected ExecutorService workFlowPool;
    protected TaskFlowConfig taskFlowConfig;
    protected List<RecommendConfig.Chain> chains = Lists.newArrayList();
    protected Set<String> processedTask = Sets.newHashSet();
    protected Map<String, DataService> taskServices;

    protected TaskFlow taskFlow;

    public boolean init(String name, TaskFlowConfig taskFlowConfig, TaskFlow taskFlow, ExecutorService workFlowPool) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null or taskServices is null , init fail!");
            return false;
        }
        this.name = name;
        this.workFlowPool = workFlowPool;
        this.taskServices = taskFlow.getTaskServices();
        this.taskFlowConfig = taskFlowConfig;
        this.taskFlow = taskFlow;
        return initService();
    }
    protected abstract boolean initService();

    public boolean checkRequest(ServiceRequest request, DataContext context) {
        if (request != null && request.isCircular()) {
            log.error("service in circular dependency");
            throw new RuntimeException("service in circular dependency");
        }
        return true;
    }

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
                (result.getResultType() == ResultTypeEnum.DATA && result.getData() == null)) {
            log.warn("result set value and data is null!");
            return false;
        }
        return true;
    }

    public DataResult getDataResultByName(String name, DataContext context) {
        if (context.getStatus(name) != TaskStatusEnum.SUCCESS) {
            log.error("name ：{} result get fail！", name);
            return null;
        }
        DataResult result = context.getResult(name);
        if (result == null || !result.isVaild()) {
            log.error("name ：{} result get wrong！", name);
            return null;
        }
        return result;
    }

    public List<DataResult> getDataResultByNames(List<String> names, DataContext context) {
        List<DataResult> dataResults = Lists.newArrayList();
        for (String name : names) {
            DataResult result = getDataResultByName(name, context);
            if (result == null) {
                log.error("name ：{} result get wrong！", name);
                continue;
            }
            dataResults.add(result);
        }
        return dataResults;
    }

    public List<Map> getTaskResultByColumns(List<String> taskNames, boolean isAny, List<String> columnNames, DataContext context) {
        List<DataResult> dataResults = getDataResultByNames(taskNames, context);
        List<Map> data = Lists.newArrayList();
        if ((!isAny && dataResults.size() != taskNames.size()) || (isAny && dataResults.isEmpty())) {
            log.error("TaskResult：{} result get wrong！", taskNames);
            return data;
        }
        for (DataResult dataResult: dataResults) {
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
            }
        }
        return data;
    }

    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        if (StringUtils.isEmpty(depend)) {
            depend = name;
        }
        ServiceRequest req = new ServiceRequest(depend, name);
        req.copy(request);
        if (context != null && MapUtils.isNotEmpty(context.getRequest())) {
            if (req.getData() == null) req.setData(Maps.newHashMap());
            req.getData().putAll(context.getRequest());
        }
        return req;
    }

    public ServiceRequest makeRequest(String depend, DataContext context) {
        return makeRequest(depend, null, context);
    }

    public DataResult execute(DataContext context) {
        return execute(name, makeRequest(name, context), context);
    }

    public DataResult execute(String depend, DataContext context) {
        return execute(depend, makeRequest(depend, context), context);
    }
    public DataResult execute(String depend, ServiceRequest request, DataContext context) {
        DataService dataService = taskServices.get(depend);
        if (dataService == null) {
            log.error("task:{} depend:{} service init fail!", name, depend);
            context.setStatus(name, TaskStatusEnum.DEPEND_INIT_FAIL);
            return null;
        }
        ServiceRequest taskRequest = makeRequest(depend, request, context);
        if (taskRequest == null) {
            log.error("task:{} depend:{} request init fail!", name, depend);
            context.setStatus(name, TaskStatusEnum.DEPEND_INIT_FAIL);
            return null;
        }
        dataService.execute(taskRequest, context);
        processedTask.add(depend);
        return getDataResultByName(depend, context);
    }

    public List<DataResult> execute(List<String> names, boolean isWhen, boolean isAny, ServiceRequest request, DataContext context) {
        RecommendConfig.Chain chain = new RecommendConfig.Chain();
        if (isWhen) {
            chain.setWhen(names);
            chain.setAny(isAny);
        } else {
            chain.setThen(names);
        }
        ServiceRequest req = makeRequest(null, request, context);
        List<String> outputs = executeChain(chain, req, context);
        return getDataResultByNames(outputs, context);
    }

    public List<DataResult> execute(List<String> names, boolean isWhen, boolean isAny, DataContext context) {
        return execute(names, isWhen, isAny, null, context);
    }

    public List<DataResult> execute(List<String> names, DataContext context) {
        return execute(names, false, false, context);
    }

    public List<String> executeChain(RecommendConfig.Chain chain, DataContext context) {
        return executeChain(chain, makeRequest(name, context), context);
    }

    public List<String> executeChain(RecommendConfig.Chain chain, ServiceRequest request, DataContext context) {
        List<String> result = Lists.newArrayList();
        String name = chain.getName();
        String lastThenTask = null;
        if (CollectionUtils.isNotEmpty(chain.getThen())) {
            for (String taskName : chain.getThen()) {
                lastThenTask = taskName;
                if (execute(taskName, request, context) == null) {
                    log.error("task:{} depend:{} exec fail status:{}!", name, taskName, context.getStatus(taskName).getName());
                    context.setStatus(name, TaskStatusEnum.DEPEND_EXEC_FAIL);
                    return result;
                }
            }
        }
        if (CollectionUtils.isNotEmpty(chain.getWhen())) {
            List<CompletableFuture<?>> whenList = Lists.newArrayList();
            for (String taskName : chain.getWhen()) {
                whenList.add(CompletableFuture.supplyAsync(() -> execute(taskName, request, context), workFlowPool)
                        .whenComplete(((dataResult, throwable) -> {
                            if (context.getStatus(taskName) != TaskStatusEnum.SUCCESS) {
                                log.error("task:{} depend:{} exec fail status:{}!", name, taskName, context.getStatus(taskName).getName());
                                context.setStatus(name, TaskStatusEnum.DEPEND_EXEC_FAIL);
                            }
                            if (throwable != null) {
                                log.error("exception:{}", throwable.getMessage());
                                context.setStatus(name, TaskStatusEnum.DEPEND_EXEC_FAIL);
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
                context.setStatus(name, TaskStatusEnum.EXEC_FAIL);
            } catch (TimeoutException e) {
                log.error("when task timeout!", e);
                context.setStatus(name, TaskStatusEnum.EXEC_FAIL);
            }
            result.addAll(chain.getWhen());
        } else if (StringUtils.isNotEmpty(lastThenTask)) {
            result.add(lastThenTask);
        }
        return result;
    }

    public DataResult execute(ServiceRequest request, DataContext context){
        DataResult result = new DataResult();
        boolean reset = request.get("reset", false);
        String reqSign = context.genRequestSign(request);
        if (!reset && context.getStatus(name) == TaskStatusEnum.SUCCESS) {
            result = context.getResult(name);
            if (result != null && result.isVaild() && result.getReqSign().equals(reqSign)) {
                return result;
            }
        }
        context.setStatus(name, TaskStatusEnum.INIT);
        if (StringUtils.isEmpty(request.getName()) || !name.equals(request.getName())) {
            request.setName(name);
        }
        if (!checkRequest(request, context)) {
            log.error("request in task:{} is check fail!", name);
            context.setStatus(name, TaskStatusEnum.CHECK_FAIL);
            return result;
        }

        for (RecommendConfig.Chain chain : chains) {
            List<String> taskNames = executeChain(chain, request, context);
            if (StringUtils.isNotEmpty(chain.getName())) {
                if (CollectionUtils.isEmpty(taskNames)) {
                    log.error("task:{} chain:{} no task!", name, chain.getName());
                    context.setStatus(chain.getName(), TaskStatusEnum.DEPEND_INIT_FAIL);
                    return result;
                }
                DataResult chainResult = new DataResult();
                List<Map> data = getTaskResultByColumns(taskNames, chain.isAny(), chain.getColumnNames(), context);
                if (data == null) {
                    log.error("task:{} chain:{} get result fail!", name, chain.getName());
                    context.setStatus(chain.getName(), TaskStatusEnum.EXEC_FAIL);
                    return result;
                }
                chainResult.setData(data);
                context.setResult(chain.getName(), chainResult);
                context.setStatus(chain.getName(), TaskStatusEnum.SUCCESS);
            }
        }
        if (context.getStatus(name) != TaskStatusEnum.INIT) {
            return result;
        }
        result = process(request, context);
        if (checkResult(result)) {
            result.setReqSign(reqSign);
            context.setResult(name, result);
            context.setStatus(name, TaskStatusEnum.SUCCESS);
        } else {
            context.setStatus(name, TaskStatusEnum.RESULT_ERROR);
        }
        return result;
    }

    protected abstract DataResult process(ServiceRequest request, DataContext context);
}
