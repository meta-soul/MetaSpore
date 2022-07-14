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
        LettuceConnectionFactory factory;
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
    public boolean checkRequest(ServiceRequest request, DataContext context) {
        List<String> keys = request.getKeys();
        if (CollectionUtils.isEmpty(keys)) {
            log.error("redis request keys must not be empty!");
            return false;
        }
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = new DataResult();
        String parent = request.getParent();
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(parent);
        List<String> columnNames = sourceTable.getColumnNames();
        List<String> keys = request.getKeys();
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        int limit = request.getLimit();
        List<Map> data = Lists.newArrayList();
        if (source.getOptions().getOrDefault("type", "hash").equals("hash")) {
            for (String item : keys) {
                Map value = redisTemplate.opsForHash().entries(item);
                if (MapUtils.isEmpty(value)) {
                    continue;
                }
                Map<String, Object> map = Maps.newHashMap();
                map.put(columnNames.get(0), item);
                map.put(columnNames.get(1), value);
                data.add(map);
            }
        }  else if (source.getOptions().getOrDefault("type", "hash").equals("value")) {
            for (String item : keys) {
                Object value = redisTemplate.opsForValue().get(String.valueOf(item));
                if (value == null) {
                    continue;
                }
                Map<String, Object> map = Maps.newHashMap();
                map.put(columnNames.get(0), item);
                map.put(columnNames.get(1), value);
                data.add(map);
            }
        } else if (source.getOptions().getOrDefault("type", "hash").equals("list")) {
            for (String item : keys) {
                List value = redisTemplate.opsForList().range(String.valueOf(item), 0, limit);
                if (CollectionUtils.isEmpty(value)) {
                    continue;
                }
                Map<String, Object> map = Maps.newHashMap();
                map.put(columnNames.get(0), item);
                map.put(columnNames.get(1), value);
                data.add(map);
            }
        } else if (source.getOptions().getOrDefault("type", "hash").equals("set")) {
            for (String item : keys) {
                Set value = redisTemplate.opsForSet().members(String.valueOf(item));
                if (CollectionUtils.isEmpty(value)) {
                    continue;
                }
                Map<String, Object> map = Maps.newHashMap();
                map.put(columnNames.get(0), item);
                map.put(columnNames.get(1), value);
                data.add(map);
            }
        } else if (source.getOptions().getOrDefault("type", "hash").equals("zset")) {
            for (String item : keys) {
                Set value = redisTemplate.opsForZSet().range(String.valueOf(item), 0, limit);
                if (CollectionUtils.isEmpty(value)) {
                    continue;
                }
                Map<String, Object> map = Maps.newHashMap();
                map.put(columnNames.get(0), item);
                map.put(columnNames.get(1), value);
                data.add(map);
            }
        }
        result.setData(data);
        return result;
    }
}
