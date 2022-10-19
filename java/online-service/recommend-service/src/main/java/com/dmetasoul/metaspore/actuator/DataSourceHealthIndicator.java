package com.dmetasoul.metaspore.actuator;

import com.dmetasoul.metaspore.baseservice.TaskServiceRegister;
import com.dmetasoul.metaspore.datasource.DataSource;
import com.dmetasoul.metaspore.enums.HealthStatus;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.Status;
import org.springframework.cloud.context.config.annotation.RefreshScope;

import org.springframework.stereotype.Component;

import java.util.Map;

@Slf4j
@RefreshScope
@Component
public class DataSourceHealthIndicator extends AbstractHealthIndicator {

    @Autowired
    private TaskServiceRegister taskServiceRegister;

    @Override
    protected void doHealthCheck(Health.Builder builder) throws Exception {
        if (taskServiceRegister == null || MapUtils.isEmpty(taskServiceRegister.getDataSources())) {
            builder.status(String.valueOf(HealthStatus.OUT_OF_SERVICE)).withDetail("message", "no found DataSources");
            return;
        }
        Map<String, Object> details = Maps.newHashMap();
        HealthStatus status = HealthStatus.UP;
        Throwable exception = null;
        for (Map.Entry<String, DataSource> entry : taskServiceRegister.getDataSources().entrySet()) {
            HealthStatus sourceStatus = HealthStatus.UP;
            Map<String, Object> sourceDetails = Maps.newHashMap();
            entry.getValue().doHealthCheck(sourceStatus, sourceDetails, exception);
            sourceDetails.put("status", sourceStatus);
            details.put(entry.getKey(), sourceDetails);
        }
        builder.status(String.valueOf(status)).withDetails(details);
    }


}
