package com.dmetasoul.metaspore.actuator;

import java.util.Collection;
import java.util.Set;
import org.springframework.boot.actuate.endpoint.annotation.Endpoint;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.cloud.context.refresh.ContextRefresher;
import org.springframework.stereotype.Component;

@Endpoint(
        id = "pullConfig"
)
@Component
public class PullConfigEndpoint {
    private ContextRefresher contextRefresher;

    public PullConfigEndpoint(ContextRefresher contextRefresher) {
        this.contextRefresher = contextRefresher;
    }

    @ReadOperation
    public Collection<String> pull() {
        Set<String> keys = this.contextRefresher.refresh();
        return keys;
    }
}

