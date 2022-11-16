package com.dmetasoul.metaspore.configure;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Slf4j
@Data
@Component
@ConfigurationProperties(prefix = "service.configure")
public class ServiceConfig {
    String initModelInfo;
    String initConfig;
    String initConfigFormat;
}
