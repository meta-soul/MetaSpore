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
package com.dmetasoul.metaspore.recommend.taskService;

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.dataservice.DataService;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("Scene")
public class SceneTask extends TaskService {
    private RecommendConfig.Scene scene;

    @Override
    public boolean initService() {
        scene = taskFlowConfig.getScenes().get(name);
        chains = scene.getChains();
        return true;
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        ServiceRequest req = super.makeRequest(depend, request, context);
        if (MapUtils.isNotEmpty(scene.getOptions())) {
            req.getData().putAll(scene.getOptions());
        }
        return req;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult dataResult = null;
        RecommendConfig.Chain chain = chains.get(chains.size() - 1);
        List<String> outputs = chain.getWhen();
        boolean isAny = false;
        if (CollectionUtils.isEmpty(outputs)) {
            if (CollectionUtils.isEmpty(chain.getThen())) {
                return dataResult;
            }
            int lastIndex = chain.getThen().size() - 1;
            outputs = List.of(chain.getThen().get(lastIndex));
        } else {
            isAny = chain.isAny();
        }
        dataResult = new DataResult();
        List<Map> data = getTaskResultByColumns(outputs, isAny, scene.getColumnNames(), context);
        if (data == null) {
            log.error("scene:{} last chain task:{} get result fail!", name, outputs);
            context.setStatus(name, TaskStatusEnum.EXEC_FAIL);
            return null;
        }
        dataResult.setData(data);
        return dataResult;
    }
}
