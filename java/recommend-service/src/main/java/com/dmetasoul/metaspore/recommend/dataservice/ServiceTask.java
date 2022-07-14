package com.dmetasoul.metaspore.recommend.dataservice;

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.recommend.RecommendService;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("Service")
public class ServiceTask extends DataService {

    private RecommendService recommendService;

    private RecommendConfig.Service service;

    @Override
    public boolean initService() {
        service = taskFlowConfig.getServices().get(name);
        recommendService = taskFlow.getRecommendService(service);
        if (recommendService == null) {
            log.error("recommendService:{} is init fail in {}", service.getServiceName(), name);
            return false;
        }
        recommendService.init(name, taskFlowConfig, this);
        return true;
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
