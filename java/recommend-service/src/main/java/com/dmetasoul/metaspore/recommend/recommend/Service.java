package com.dmetasoul.metaspore.recommend.recommend;

import com.dmetasoul.metaspore.recommend.baseservice.TaskServiceRegister;
import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.configure.TransformConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.dataservice.DataService;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.Assert;

import java.util.List;
import java.util.concurrent.*;

@Slf4j
@Data
@ServiceAnnotation("Service")
public class Service extends Transform implements BaseService {
    protected String name;
    protected ExecutorService taskPool;
    protected TaskFlowConfig taskFlowConfig;
    protected List<DataService> tasks;
    protected TaskServiceRegister serviceRegister;
    protected RecommendConfig.Service serviceConfig;
    protected long timeout = 30000L;
    protected TimeUnit timeUnit = TimeUnit.MILLISECONDS;

    public boolean init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister serviceRegister) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null, init fail!");
            return false;
        }
        this.name = name;
        this.taskFlowConfig = taskFlowConfig;
        this.serviceRegister = serviceRegister;
        this.serviceConfig = taskFlowConfig.getServices().get(name);
        this.taskPool = serviceRegister.getTaskPool();
        if (CollectionUtils.isNotEmpty(serviceConfig.getColumnNames())) {
            resFields = Lists.newArrayList();
            dataTypes = Lists.newArrayList();
            for (String col : serviceConfig.getColumnNames()) {
                resFields.add(serviceConfig.getFieldMap().get(col));
                dataTypes.add(serviceConfig.getColumnMap().get(col));
            }
        }
        super.initTransform(name, taskPool, serviceRegister);
        return initService();
    }

    protected boolean initService() {
        timeout = CommonUtils.getField(serviceConfig.getOptions(), "timeout", timeout);
        tasks = Lists.newArrayList();
        if (CollectionUtils.isNotEmpty(serviceConfig.getTasks())) {
            for (String item : serviceConfig.getTasks()) {
                DataService task = serviceRegister.getDataService(item);
                tasks.add(task);
            }
        }
        return true;
    }

    public DataTypeEnum getFieldType(String key) {
        return serviceConfig.getColumnMap().get(key);
    }

    public void initFunctions() {
    }
    @SneakyThrows
    protected CompletableFuture<List<DataResult>> executeTask(List<DataResult> data, DataContext context) {
        List<CompletableFuture<DataResult>> taskList = Lists.newArrayList();
        for (DataService task : tasks) {
            if (CollectionUtils.isNotEmpty(data)) {
                for (DataResult item : data) {
                    if (StringUtils.isNotEmpty(item.getName())) {
                        task.setDataResultByName(item.getName(), item, context);
                    }
                }
            }
            taskList.add(CompletableFuture.supplyAsync(() -> task.execute(context), taskPool)
                    .whenComplete(((dataResult, throwable) -> {
                        if (throwable != null) {
                            log.error("exception:{}", throwable.getMessage());
                        }
                    }))
            );
        }
        return CompletableFuture.allOf(taskList.toArray(new CompletableFuture[]{}))
                .thenApplyAsync(x -> {
                    List<DataResult> result = Lists.newArrayList();
                    for (CompletableFuture<DataResult> future : taskList) {
                        try {
                            result.add(future.get(timeout, timeUnit));
                        } catch (InterruptedException | ExecutionException | TimeoutException e) {
                            throw new RuntimeException(e);
                        }
                    }
                    return result;
                }, taskPool);
    }
    @Override
    public CompletableFuture<List<DataResult>> execute(List<DataResult> data, DataContext context) {
        CompletableFuture<List<DataResult>> future = CompletableFuture.supplyAsync(() -> data);
        if (CollectionUtils.isNotEmpty(serviceConfig.getPreTransforms())) {
            future = executeTransform(future, serviceConfig.getPreTransforms(), serviceConfig.getOptions(), context);
            Assert.notNull(future, "Service execute pre-transform function fail at " + name);
        }
        future = future.thenApplyAsync(dataResults -> {
            List<DataResult> result = Lists.newArrayList();
            if (CollectionUtils.isNotEmpty(dataResults)) {
                result.addAll(dataResults);
            }
            List<DataResult> list;
            try {
                list = executeTask(result, context).get(timeout, timeUnit);
            } catch (InterruptedException | ExecutionException | TimeoutException e) {
                log.error("service exception e: {}", e.getMessage());
                throw new RuntimeException(e);
            }
            return list;
        });
        if (CollectionUtils.isNotEmpty(serviceConfig.getTransforms())) {
            future = executeTransform(future, serviceConfig.getTransforms(), serviceConfig.getOptions(), context);
            Assert.notNull(future, "Service execute transform function fail at " + name);
        }
        if (!hasSomeTransform(serviceConfig.getTransforms(), "cutOff")
                && serviceConfig.getOptions().containsKey("maxReservation")) {
            TransformConfig transformConfig = new TransformConfig();
            transformConfig.setName("cutOff");
            future = executeTransform(future, List.of(transformConfig), serviceConfig.getOptions(), context);
            Assert.notNull(future, "Service execute transform function fail in cutoff at " + name);
        }
        return future;
    }

    @Override
    public CompletableFuture<List<DataResult>> execute(DataResult data, DataContext context) {
        return execute(List.of(data), context);
    }

    @Override
    public CompletableFuture<List<DataResult>> execute(DataContext context) {
        Assert.isTrue(CollectionUtils.isNotEmpty(serviceConfig.getTasks()), "executeTask tasks must not empty");
        return execute(List.of(), context);
    }

    @Override
    public void close() {
    }
}
