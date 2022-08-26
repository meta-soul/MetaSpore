//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
package com.dmetasoul.metaspore.recommend.datasource;

import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.springframework.boot.actuate.health.Status;
import org.springframework.data.redis.connection.*;
import org.springframework.data.redis.connection.lettuce.LettuceClientConfiguration;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.data.redis.connection.lettuce.LettucePoolingClientConfiguration;
import org.springframework.data.redis.core.RedisConnectionUtils;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.Jackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;
import org.springframework.util.Assert;

import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;
/**
 * source.kind=redis的DataSource实现类
 * 配置中的source.kind需要与注解DataSourceAnnotation中value保持一致
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@SuppressWarnings("rawtypes")
@Data
@Slf4j
@ServiceAnnotation("Redis")
public class RedisSource extends DataSource {

    private RedisTemplate<String, Object> redisTemplate;
    private LettuceConnectionFactory factory;

    private FeatureConfig.Source source;

    private static RedisNode readHostAndPortFromString(String hostAndPort) {
        String[] args = StringUtils.split(hostAndPort, ":");
        Assert.notNull(args, "HostAndPort need to be seperated by  ':'.");
        Assert.isTrue(args.length == 2, "Host and Port String needs to specified as host:port");
        return new RedisNode(args[0], Integer.parseInt(args[1]));
    }

    public RedisTemplate<String, Object> getRedisTemplate(RedisConnectionFactory factory) {
        @SuppressWarnings("unchecked") Jackson2JsonRedisSerializer jackson2JsonRedisSerializer = new Jackson2JsonRedisSerializer(Object.class);
        ObjectMapper om = new ObjectMapper();
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
        //noinspection deprecation
        om.enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL);
        jackson2JsonRedisSerializer.setObjectMapper(om);
        RedisTemplate<String, Object> redisTemplate = new RedisTemplate<>();
        redisTemplate.setConnectionFactory(factory);
        StringRedisSerializer stringSerializer = new StringRedisSerializer();

        redisTemplate.setKeySerializer(stringSerializer);
        redisTemplate.setValueSerializer(jackson2JsonRedisSerializer);

        redisTemplate.setHashKeySerializer(stringSerializer);
        redisTemplate.setHashValueSerializer(jackson2JsonRedisSerializer);
        redisTemplate.afterPropertiesSet();
        return redisTemplate;
    }

    public GenericObjectPoolConfig genericObjectPoolConfig() {
        Map<String, Object> lettucePoolInfo = Utils.getField(source.getOptions(), "lettuce-pool", Maps.newHashMap());
        //连接池配置
        GenericObjectPoolConfig genericObjectPoolConfig =
                new GenericObjectPoolConfig();
        genericObjectPoolConfig.setMaxIdle(Utils.getField(lettucePoolInfo, "max-idle", 10));
        genericObjectPoolConfig.setMinIdle(Utils.getField(lettucePoolInfo, "min-idle", 1));
        genericObjectPoolConfig.setMaxTotal(Utils.getField(lettucePoolInfo, "max-active", 10));
        genericObjectPoolConfig.setMaxWaitMillis(Utils.getField(lettucePoolInfo, "max-wait", 10000));
        return genericObjectPoolConfig;
    }
    public RedisStandaloneConfiguration getRedisStandaloneConfiguration(GenericObjectPoolConfig genericObjectPoolConfig, Map<String, Object> redisConfig) {
        String host = Utils.getField(redisConfig,"host", "localhost");
        int port = Utils.getField(redisConfig,"port", 6379);
        String password = Utils.getField(redisConfig,"password", "");
        RedisStandaloneConfiguration redisStandaloneConfiguration = new RedisStandaloneConfiguration(host,port);
        if (StringUtils.isNotEmpty(password)) {
            redisStandaloneConfiguration.setPassword(password);
        }
        return redisStandaloneConfiguration;
    }

    public RedisClusterConfiguration getRedisClusterConfiguration(GenericObjectPoolConfig genericObjectPoolConfig, Map<String, Object> redisConfig) {
        String nodes = Utils.getField(redisConfig,"nodes", "localhost:6379");
        int maxRedirects = Utils.getField(redisConfig,"max-redirect", 1);
        String password = Utils.getField(redisConfig,"password", "");
        Set<RedisNode> clusterNodes = Arrays.stream(nodes.split(",")).map(RedisSource::readHostAndPortFromString).collect(Collectors.toSet());
        RedisClusterConfiguration configuration = new RedisClusterConfiguration();
        configuration.setClusterNodes(clusterNodes);
        configuration.setMaxRedirects(maxRedirects);
        if (StringUtils.isNotEmpty(password)) {
            configuration.setPassword(password);
        }
        return configuration;
    }

    public RedisSentinelConfiguration getRedisSentinelConfiguration(GenericObjectPoolConfig genericObjectPoolConfig, Map<String, Object> redisConfig) {
        String master = Utils.getField(redisConfig,"master", "myMaster");
        String nodes = Utils.getField(redisConfig,"nodes", "localhost:6379");
        String password = Utils.getField(redisConfig,"password", "");
        Set<String> clusterNodes = Arrays.stream(nodes.split(",")).collect(Collectors.toSet());
        RedisSentinelConfiguration redisSentinelConfiguration = new RedisSentinelConfiguration(master, clusterNodes);
        if (StringUtils.isNotEmpty(password)) {
            redisSentinelConfiguration.setPassword(password);
        }
        return redisSentinelConfiguration;
    }

    @Override
    public boolean initService() {
        source = taskFlowConfig.getSources().get(name);
        if (!source.getKind().equalsIgnoreCase("redis")) {
            log.error("config redis fail! is not kind:{} eq redis!", source.getKind());
            return false;
        }
        GenericObjectPoolConfig genericObjectPoolConfig = genericObjectPoolConfig();
        LettuceClientConfiguration clientConfig = LettucePoolingClientConfiguration.builder()
                .commandTimeout(Duration.ofMillis(Utils.getField(source.getOptions(), "timeout", 10000)))
                .poolConfig(genericObjectPoolConfig)
                .build();

        Map<String, Object> redisConfig = Utils.getField(source.getOptions(), "standalone", Maps.newHashMap());
        if (MapUtils.isNotEmpty(redisConfig)) {
            RedisStandaloneConfiguration configuration = getRedisStandaloneConfiguration(genericObjectPoolConfig, redisConfig);
            factory = new LettuceConnectionFactory(configuration, clientConfig);
        }
        if (factory == null) {
            redisConfig = Utils.getField(source.getOptions(), "sentinel", Maps.newHashMap());
            if (MapUtils.isNotEmpty(redisConfig)) {
                RedisSentinelConfiguration configuration = getRedisSentinelConfiguration(genericObjectPoolConfig, redisConfig);
                factory = new LettuceConnectionFactory(configuration, clientConfig);
            }
        }
        if (factory == null) {
            redisConfig = Utils.getField(source.getOptions(), "cluster", Maps.newHashMap());
            if (MapUtils.isNotEmpty(redisConfig)) {
                RedisClusterConfiguration configuration = getRedisClusterConfiguration(genericObjectPoolConfig, redisConfig);
                factory = new LettuceConnectionFactory(configuration, clientConfig);
            }
        }
        Assert.notNull(factory, "redis LettuceConnectionFactory init fail");
        factory.afterPropertiesSet();
        redisTemplate = getRedisTemplate(factory);
        return true;
    }

    @Override
    public void close() {
        if (factory != null) {
            try {
                factory.destroy();
            } catch (Exception ex) {
                log.error("redis factory destroy fail! {}", ex.getMessage());
            }
        }
    }
    @Override
    public void doHealthCheck(Status status, Map<String, Object> details, Throwable exception) throws Exception {
        super.doHealthCheck(status, details, exception);
        RedisConnection connection = RedisConnectionUtils.getConnection(factory);
        try {
            if (connection instanceof RedisClusterConnection) {
                ClusterInfo clusterInfo = ((RedisClusterConnection)connection).clusterGetClusterInfo();
                details.put("cluster_size", clusterInfo.getClusterSize());
                details.put("slots_up", clusterInfo.getSlotsOk());
                details.put("slots_fail", clusterInfo.getSlotsFail());
                if ("fail".equalsIgnoreCase(clusterInfo.getState())) {
                    status = Status.DOWN;
                }
            } else {
                details.put("version", Objects.requireNonNull(connection.info("server")).getProperty("redis_version"));
            }
        } finally {
            RedisConnectionUtils.releaseConnection(connection, factory);
        }
    }
}
