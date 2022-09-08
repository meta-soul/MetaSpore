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
package com.dmetasoul.metaspore.recommend.data;

import com.google.common.collect.Maps;
import com.google.gson.Gson;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.commons.lang3.StringUtils;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import static com.dmetasoul.metaspore.recommend.common.Utils.genResultKey;
import static com.dmetasoul.metaspore.recommend.common.Utils.getObjectToMap;
/**
 * 用于服务请求中的上下文数据
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
public class DataContext {
    private String id;
    private Map<String, Object> request;
    private ConcurrentMap<String, DataResult> results = Maps.newConcurrentMap();

    public DataContext() {
        this.id = UUID.randomUUID().toString().replaceAll("-", "");
    }

    public DataContext(String id) {
        this.id = id;
    }

    public DataContext(Map<String, Object> request) {
        this.request = request;
    }

    public void setRequest(Object req) {
        if (req != null) {
            request = getObjectToMap(req);
        }
    }

    public void setRequest(String req) {
        if (StringUtils.isNotEmpty(req)) {
            request = new Gson().fromJson(req, Map.class);
        }
    }

    public void setRequest(Map<String, Object> req) {
        request = req;
    }

    public DataResult getResult(String name) {
        return results.get(name);
    }

    public void setResult(String name, DataResult result) {
        results.put(name, result);
    }

    public DataResult getResult(String name, String taskName) {
        String key = genResultKey(name, taskName);
        return results.get(key);
    }
    public void setResult(String name, String taskName, DataResult result) {
        String key = genResultKey(name, taskName);
        results.put(key, result);
    }

}
