package com.dmetasoul.metaspore.actuator;

import org.springframework.boot.autoconfigure.AutoConfigureBefore;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.autoconfigure.RefreshAutoConfiguration;
import org.springframework.cloud.context.scope.refresh.RefreshScope;
import org.springframework.cloud.util.ConditionalOnBootstrapEnabled;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration(
        proxyBeanMethods = false
)
@ConditionalOnClass({RefreshScope.class})
@ConditionalOnProperty(
        name = {"spring.cloud.pullConfig.enabled"},
        matchIfMissing = true
)
@AutoConfigureBefore({HibernateJpaAutoConfiguration.class})
@EnableConfigurationProperties({RefreshAutoConfiguration.RefreshProperties.class})
public class PullConfigAutoConfiguration {
    public static final String PULLCONFIG_SCOPE_NAME = "pullConfig";
    public static final String PULLCONFIG_SCOPE_PREFIX = "spring.cloud.pullConfig";
    public static final String PULLCONFIG_SCOPE_ENABLED = "spring.cloud.pullConfig.enabled";

    public PullConfigAutoConfiguration() {
    }

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnBootstrapEnabled
    public PullContextRefresher pullContextRefresher(ConfigurableApplicationContext context, RefreshScope scope, RefreshAutoConfiguration.RefreshProperties properties) {
        return new PullContextRefresher(context, scope, properties);
    }

}