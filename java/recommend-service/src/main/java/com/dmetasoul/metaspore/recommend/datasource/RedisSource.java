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

import com.dmetasoul.metaspore.recommend.annotation.DataSourceAnnotation;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.data.redis.connection.RedisClusterConfiguration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.RedisNode;
import org.springframework.data.redis.connection.RedisStandaloneConfiguration;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.Jackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;
import org.springframework.util.Assert;

import java.util.*;
import java.util.stream.Collectors;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataSourceAnnotation("redis")
public class RedisSource extends DataSource {

    private RedisTemplate<String, Object> redisTemplate;
    private LettuceConnectionFactory factory;

    private static RedisNode readHostAndPortFromString(String hostAndPort) {
        String[] args = StringUtils.split(hostAndPort, ":");
        Assert.notNull(args, "HostAndPort need to be seperated by  ':'.");
        Assert.isTrue(args.length == 2, "Host and Port String needs to specified as host:port");
        return new RedisNode(args[0], Integer.valueOf(args[1]));
    }

    public RedisTemplate<String, Object> getRedisTemplate(RedisConnectionFactory factory) {
        Jackson2JsonRedisSerializer jackson2JsonRedisSerializer = new Jackson2JsonRedisSerializer(Object.class);
        ObjectMapper om = new ObjectMapper();
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
        om.enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL);
        jackson2JsonRedisSerializer.setObjectMapper(om);
        RedisTemplate<String, Object> redisTemplate = new RedisTemplate<String, Object>();
        redisTemplate.setConnectionFactory(factory);
        StringRedisSerializer stringSerializer = new StringRedisSerializer();

        redisTemplate.setKeySerializer(stringSerializer);
        redisTemplate.setValueSerializer(jackson2JsonRedisSerializer);

        redisTemplate.setHashKeySerializer(stringSerializer);
        redisTemplate.setHashValueSerializer(jackson2JsonRedisSerializer);
        redisTemplate.afterPropertiesSet();
        return redisTemplate;
    }

    @Override
    public boolean initService() {
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        if (!source.getKind().equals("redis")) {
            log.error("config redis fail! is not kind:{} eq redis!", source.getKind());
            return false;
        }
        boolean isCluster = (Boolean) source.getOptions().getOrDefault("cluster", true);
        if (isCluster) {
            String nodes = (String) source.getOptions().getOrDefault("nodes", "localhost:6379");
            Set<RedisNode> clusterNodes = Arrays.stream(nodes.split(",")).map(RedisSource::readHostAndPortFromString).collect(Collectors.toSet());
            RedisClusterConfiguration configuration = new RedisClusterConfiguration();
            configuration.setClusterNodes(clusterNodes);
            factory = new LettuceConnectionFactory(configuration);
        } else {
            String host = (String) source.getOptions().getOrDefault("host", "localhost");
            int port = (Integer) source.getOptions().getOrDefault("port", 6379);
            RedisStandaloneConfiguration configuration = new RedisStandaloneConfiguration();
            configuration.setHostName(host);
            configuration.setPort(port);
            factory = new LettuceConnectionFactory(configuration);
        }
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
}
