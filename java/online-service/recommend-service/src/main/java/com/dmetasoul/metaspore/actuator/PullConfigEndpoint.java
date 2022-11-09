package com.dmetasoul.metaspore.actuator;

import java.util.Collection;
import java.util.Set;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuate.endpoint.annotation.Endpoint;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.stereotype.Component;

@Endpoint(
        id = "pullConfig"
)
@Component
@Slf4j
public class PullConfigEndpoint {
    @Autowired
    private PullContextRefresher pullContextRefresher;

    @ReadOperation
    public Collection<String> pull() {
        Set<String> keys = this.pullContextRefresher.pullConfig();
        return keys;
    }
}

