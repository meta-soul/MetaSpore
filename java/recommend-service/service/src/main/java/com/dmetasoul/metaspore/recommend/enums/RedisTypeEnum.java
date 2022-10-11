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
package com.dmetasoul.metaspore.recommend.enums;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ZSetOperations;

import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
public enum RedisTypeEnum {
    VALUE(0, "value", new RedisOpsFor<>() {
        @Override
        protected Object ops(String key, int limit) {
            return redisTemplate.opsForValue().get(getRedisKey(key));
        }

        @Override
        protected List<Map<String, Object>> process(String key, int limit) {
            return List.of(getMap(key, ops(key, limit)));
        }
    }),
    HASH(1,"hash", new RedisOpsFor<Map<String, Object>>(){
        @Override
        protected Map ops(String key, int limit) {
            return redisTemplate.opsForHash().entries(getRedisKey(key));
        }

        @Override
        protected List<Map<String, Object>> process(String key, int limit) {
            Map<String, Object> map = getMap(key, ops(key, limit));
            map.put(columnNames.get(0), key);
            for (int i = 1; i < columnNames.size(); ++i) {
                String col = columnNames.get(i);
                map.put(col, getField(col, map));
            }
            return List.of(map);
        }
    }),
    LIST(2,"list", new RedisOpsFor<List>(){
        @Override
        protected List ops(String key, int limit) {
            return redisTemplate.opsForList().range(getRedisKey(key), 0, limit);
        }

        @Override
        protected List<Map<String, Object>> process(String key, int limit) {
            List<Map<String, Object>> data = Lists.newArrayList();
            for (Object value : ops(key, limit)) {
                data.add(getMap(key, value));
            }
            return data;
        }
    }),
    SET(3, "set", new RedisOpsFor<Set>(){
        @Override
        protected Set ops(String key, int limit) {
            return redisTemplate.opsForSet().members(getRedisKey(key));
        }

        @Override
        protected List<Map<String, Object>> process(String key, int limit) {
            List<Map<String, Object>> data = Lists.newArrayList();
            for (Object value : ops(key, limit)) {
                data.add(getMap(key, value));
            }
            return data;
        }
    }),
    ZSET(4,"zset", new RedisOpsFor<Set<ZSetOperations.TypedTuple<Object>>>(){
        @Override
        protected  Set<ZSetOperations.TypedTuple<Object>> ops(String key, int limit) {
            return redisTemplate.opsForZSet().rangeWithScores(getRedisKey(key), 0, limit);
        }

        @Override
        protected List<Map<String, Object>> process(String key, int limit) {
            List<Map<String, Object>> data = Lists.newArrayList();
            for ( ZSetOperations.TypedTuple<Object> value : ops(key, limit)) {
                Map<String, Object> map = Maps.newHashMap();
                map.put(columnNames.get(0), key);
                if (columnNames.size() > 1) {
                    map.put(columnNames.get(1), value.getValue());
                }
                if (columnNames.size() > 2) {
                    map.put(columnNames.get(2), value.getScore());
                }
                data.add(map);
            }
            return data;
        }
    });
    private Integer id;
    private String name;
    private RedisOpsFor<?> ops;

    RedisTypeEnum(int id, String name, RedisOpsFor<?> ops){
        this.id = id;
        this.name = name;
        this.ops = ops;
    }

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public List<Map<String, Object>> process(String key, int limit) {
        if (!ops.isInit) {
            throw new RuntimeException("redis ops is not init!");
        }
        return ops.process(key, limit);
    }

    public void init(String keyFormat, RedisTemplate<String, Object> redisTemplate, List<String> columnNames) {
        ops.init(keyFormat, redisTemplate, columnNames);
    }

    public void setName(String name) {
        this.name = name;
    }

    public static RedisTypeEnum getEnumByName(String name) {
        for (RedisTypeEnum e : RedisTypeEnum.values()) {
            if (e.getName().equals(name.strip().toLowerCase())) {
                return e;
            }
        }
        log.warn("unknown RedisType name :{}, default type is hash", name);
        return RedisTypeEnum.HASH;
    }

    public static RedisTypeEnum getEnumById(int id) {
        for (RedisTypeEnum e : RedisTypeEnum.values()) {
            if (e.getId() == id) {
                return e;
            }
        }
        log.warn("unknown RedisType id :{}, default type is hash", id);
        return RedisTypeEnum.HASH;
    }

    @Data
    public static abstract class RedisOpsFor<R> {
        protected String keyFormat;
        protected RedisTemplate<String, Object> redisTemplate;
        protected List<String> columnNames;

        protected boolean isInit = false;

        public void init(String keyFormat, RedisTemplate<String, Object> redisTemplate, List<String> columnNames) {
            this.keyFormat = keyFormat;
            this.redisTemplate = redisTemplate;
            this.columnNames = columnNames;
            isInit = true;
        }
        protected abstract R ops(String key, int limit);

        protected abstract List<Map<String, Object>> process(String key, int limit);

        protected String getRedisKey(String key) {
            if (StringUtils.isNotEmpty(keyFormat)) {
                return String.format(keyFormat, key);
            }
            return key;
        }
        private Object getField(String col, JsonObject jsonObject) {
            if (jsonObject == null || jsonObject.isJsonNull()) return null;
            return jsonObject.get(col);
        }

        protected Object getField(String col, Map<String, Object> map) {
            if (map == null) return null;
            return map.get(col);
        }
        public Map<String, Object> getMap(String key, Object value) {
            Map<String, Object> map = Maps.newHashMap();
            map.put(columnNames.get(0), key);
            if (columnNames.size() == 2) {
                map.put(columnNames.get(1), value);
            } else if (value instanceof String) {
                JsonObject jsonObject = (JsonObject) JsonParser.parseString((String) value);
                for (int i = 1; i < columnNames.size(); ++i) {
                    String col = columnNames.get(i);
                    map.put(col, getField(col, jsonObject));
                }
            } else {
                map.put(columnNames.get(1), value);
            }
            return map;
        }
    }

}
