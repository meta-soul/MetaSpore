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

    public abstract void close();

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
