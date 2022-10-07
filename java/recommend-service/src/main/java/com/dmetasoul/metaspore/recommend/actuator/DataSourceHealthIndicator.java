package com.dmetasoul.metaspore.recommend.actuator;

import com.dmetasoul.metaspore.recommend.baseservice.TaskServiceRegister;
import com.dmetasoul.metaspore.recommend.datasource.DataSource;
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
@Data
@RefreshScope
@Component
public class DataSourceHealthIndicator extends AbstractHealthIndicator {

    @Autowired
    private TaskServiceRegister taskServiceRegister;

    @Override
    protected void doHealthCheck(Health.Builder builder) throws Exception {
        if (taskServiceRegister == null || MapUtils.isEmpty(taskServiceRegister.getDataSources())) {
            builder.status(Status.OUT_OF_SERVICE).withDetail("message", "no found DataSources");
            return;
        }
        Map<String, Object> details = Maps.newHashMap();
        Status status = Status.UP;
        Throwable exception = null;
        for (Map.Entry<String, DataSource> entry : taskServiceRegister.getDataSources().entrySet()) {
            Status sourceStatus = Status.UP;
            Map<String, Object> sourceDetails = Maps.newHashMap();
            entry.getValue().doHealthCheck(sourceStatus, sourceDetails, exception);
            if (!sourceStatus.equals(Status.UP)) {
                status = sourceStatus;
            }
            sourceDetails.put("status", sourceStatus);
            if (exception != null) {
                sourceDetails.put("exception", exception);
            }
            details.put(entry.getKey(), sourceDetails);
        }
        builder.status(status).withDetails(details);
        if (exception != null) {
            builder.withException(exception);
        }
    }


}
