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
import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.common.SpringBeanUtil;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.dataservice.DataService;
import com.dmetasoul.metaspore.recommend.recommend.RecommendService;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("Service")
public class ServiceTask extends TaskService {
    @Autowired
    private Map<String, RecommendService> recommendServices;

    private RecommendService recommendService;

    @Override
    public boolean initService() {
        RecommendConfig.Service service = taskFlowConfig.getServices().get(name);
        recommendService = recommendServices.get(service.getServiceName());
        if (recommendService == null) {
            log.error("recommendService:{} is init fail in {}", service.getServiceName(), name);
            return false;
        }
        recommendService.init(name, taskFlowConfig, this);
        return true;
    }

    public RecommendService getRecommendService(RecommendConfig.Service service) {
        RecommendService recommendService = (RecommendService) SpringBeanUtil.getBean(service.getServiceName());
        if (recommendService == null || !recommendService.getClass().isAnnotationPresent(RecommendAnnotation.class)) {
            log.error("the RecommendService:{} load fail!", service.getServiceName());
            return null;
        }
        return recommendService;
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        ServiceRequest req = super.makeRequest(depend, request, context);
        recommendService.fillRequest(depend, req);
        return req;
    }

    @Override
    public boolean checkResult(DataResult result) {
        if (!super.checkResult(result)) {
            return false;
        }
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = recommendService.process(request, context);
        if (result.isVaild() && checkResult(result)) {
            return result;
        }
        return null;
    }
}
