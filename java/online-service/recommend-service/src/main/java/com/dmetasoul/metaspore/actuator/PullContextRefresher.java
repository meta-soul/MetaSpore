package com.dmetasoul.metaspore.actuator;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.cloud.autoconfigure.RefreshAutoConfiguration;
import org.springframework.cloud.context.environment.EnvironmentChangeEvent;
import org.springframework.cloud.context.refresh.LegacyContextRefresher;
import org.springframework.cloud.context.scope.refresh.RefreshScope;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.env.CompositePropertySource;
import org.springframework.core.env.EnumerablePropertySource;
import org.springframework.core.env.MutablePropertySources;
import org.springframework.core.env.PropertySource;

import java.util.*;

@Slf4j
public class PullContextRefresher extends LegacyContextRefresher {
    private ConfigurableApplicationContext context;
    private RefreshScope scope;
    @Deprecated
    public PullContextRefresher(ConfigurableApplicationContext context, RefreshScope scope) {
        super(context, scope);
        this.context = context;
        this.scope = scope;
    }
    public PullContextRefresher(ConfigurableApplicationContext context, RefreshScope scope, RefreshAutoConfiguration.RefreshProperties properties) {
        super(context, scope, properties);
        this.context = context;
        this.scope = scope;
    }

    public synchronized Set<String> pullConfig() {
        Map<String, Object> before = this.extract(this.context.getEnvironment().getPropertySources());
        this.updateEnvironment();
        Set<String> keys = this.changes(before, this.extract(this.context.getEnvironment().getPropertySources())).keySet();
        if (CollectionUtils.isNotEmpty(keys)) {
            this.context.publishEvent(new EnvironmentChangeEvent(this.context, keys));
            this.scope.refreshAll();
            log.info("refresh config from keys： {}", keys);
        }
        log.info("Pull config from keys： {}", keys);
        return keys;
    }

    private Map<String, Object> changes(Map<String, Object> before, Map<String, Object> after) {
        Map<String, Object> result = Maps.newHashMap();
        for (String key : before.keySet()) {
            if (!after.containsKey(key)) {
                result.put(key, null);
            }
            else if (!equal(before.get(key), after.get(key))) {
                result.put(key, after.get(key));
            }
        }
        for (String key : after.keySet()) {
            if (!before.containsKey(key)) {
                result.put(key, after.get(key));
            }
        }
        return result;
    }

    private boolean equal(Object one, Object two) {
        if (one == null && two == null) {
            return true;
        }
        if (one == null || two == null) {
            return false;
        }
        return one.equals(two);
    }

    private Map<String, Object> extract(MutablePropertySources propertySources) {
        Map<String, Object> result = Maps.newHashMap();
        List<PropertySource<?>> sources = Lists.newArrayList();
        for (PropertySource<?> source : propertySources) {
            sources.add(0, source);
        }
        for (PropertySource<?> source : sources) {
            if (!this.standardSources.contains(source.getName())) {
                extract(source, result);
            }
        }
        return result;
    }

    private void extract(PropertySource<?> parent, Map<String, Object> result) {
        if (parent instanceof CompositePropertySource) {
            try {
                List<PropertySource<?>> sources = Lists.newArrayList();
                for (PropertySource<?> source : ((CompositePropertySource) parent).getPropertySources()) {
                    sources.add(0, source);
                }
                for (PropertySource<?> source : sources) {
                    extract(source, result);
                }
            }
            catch (Exception e) {
                return;
            }
        }
        else if (parent instanceof EnumerablePropertySource) {
            for (String key : ((EnumerablePropertySource<?>) parent).getPropertyNames()) {
                result.put(key, parent.getProperty(key));
            }
        }
    }
}
