package com.dmetasoul.metaspore.recommend.configure;

import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.context.annotation.Configuration;

import java.util.List;
import java.util.Map;

@Slf4j
@Data
@Configuration
@RefreshScope
@ConfigurationProperties(prefix = "function-config")
public class FunctionConfig {
    private List<Function> functions;

    private Map<String, Function> functionMap;

    @Data
    public static class Function {
        String name;
        private Map<String, Object> options;
    }

    public boolean checkAndInit() {
        functionMap = Maps.newHashMap();
        return true;
    }
}
