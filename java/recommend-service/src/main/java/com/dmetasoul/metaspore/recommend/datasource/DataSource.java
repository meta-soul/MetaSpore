package com.dmetasoul.metaspore.recommend.datasource;

import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;

@Slf4j
@Data
public abstract class DataSource {
    protected String name;
    protected ExecutorService featurePool;
    protected TaskFlowConfig taskFlowConfig;

    public boolean init(String name, TaskFlowConfig taskFlowConfig, ExecutorService featurePool) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null, init fail!");
            return false;
        }
        this.name = name;
        this.featurePool = featurePool;
        this.taskFlowConfig = taskFlowConfig;
        return initService();
    }
    protected abstract boolean initService();

    public boolean checkRequest(ServiceRequest request, DataContext context) {
        return true;
    }


    public CompletableFuture<DataResult> execute(ServiceRequest request, DataContext context){
        String parent = request.getParent();
        context.setStatus(name, parent, TaskStatusEnum.INIT);
        return CompletableFuture.supplyAsync(() -> checkRequest(request, context))
                .thenApplyAsync((status) -> {
                    if (status) {
                        DataResult res = process(request, context);
                        if (res.isVaild()) {
                            context.setStatus(name, parent, TaskStatusEnum.SUCCESS);
                        } else {
                            context.setStatus(name, parent, TaskStatusEnum.RESULT_ERROR);
                        }
                        return res;
                    } else {
                        context.setStatus(name, parent, TaskStatusEnum.CHECK_FAIL);
                        return null;
                    }
                }, featurePool).exceptionally(error -> {
                    log.error("exec fail at {}, exception:{}!", name, error);
                    context.setStatus(name, parent, TaskStatusEnum.EXEC_FAIL);
                    return null;
                });
    }

    protected abstract DataResult process(ServiceRequest request, DataContext context);
}
