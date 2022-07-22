package com.dmetasoul.metaspore.recommend.recommend;

import com.dmetasoul.metaspore.recommend.TaskServiceRegister;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import java.util.Map;

@Slf4j
@Data
public abstract class Service {
    protected String name;
    protected TaskServiceRegister serviceRegister;

    protected TaskFlowConfig taskFlowConfig;

    protected RecommendConfig.Service serviceConfig;

    public boolean init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister serviceRegister) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null, init fail!");
            return false;
        }
        this.name = name;
        this.taskFlowConfig = taskFlowConfig;
        this.serviceRegister = serviceRegister;
        return initService();
    }

    public <T> T getOptionOrDefault(String key, T value) {
        return Utils.getField(serviceConfig.getOptions(), key, value);
    }
    public boolean setFieldFail(Map map, int index, Object value) {
        return Utils.setFieldFail(map, serviceConfig.getColumnNames(), index, value);
    }
    public String getFieldType(String key) {
        return serviceConfig.getColumnMap().get(key);
    }
    protected abstract boolean initService();
    public abstract DataResult process(ServiceRequest request, DataContext context);
}
