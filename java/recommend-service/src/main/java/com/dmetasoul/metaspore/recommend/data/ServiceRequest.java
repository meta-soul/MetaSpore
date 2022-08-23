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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
/**
 * 用于保存服务结果
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
public class ServiceRequest implements java.io.Serializable {
    private Map<String, Object> data;
    private Map<String, DataResult> intermediate;
    private int limit = 100;

    public ServiceRequest(DataContext context) {
        if (context != null&& MapUtils.isNotEmpty(context.getRequest())) {
            Map<String, Object> request = context.getRequest();
            this.limit = (int) request.getOrDefault("limit", 100);
        }
    }

    public ServiceRequest(Map<String, Object> req) {
        if (MapUtils.isNotEmpty(req)) {
            data = Maps.newHashMap();
            data.putAll(req);
        }
    }

    public ServiceRequest(ServiceRequest req, DataContext context) {
        if (context != null&& MapUtils.isNotEmpty(context.getRequest())) {
            Map<String, Object> request = context.getRequest();
            this.limit = (int) request.getOrDefault("limit", 100);
        }
        this.copy(req);
    }

    public String genRequestSign() {
        ByteArrayOutputStream output = new ByteArrayOutputStream(10240);
        try {
            ObjectOutputStream oos = new ObjectOutputStream(output);
            oos.writeObject(this);
            MessageDigest md = MessageDigest.getInstance("md5");
            byte[] data =  md.digest(output.toByteArray());
            StringBuilder sb = new StringBuilder();
            for (byte b : data) {
                sb.append(Integer.toHexString(b & 0xff));
            }
            oos.close();
            output.close();
            return sb.toString();
        } catch (IOException | NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    public void copy(ServiceRequest req) {
        if (req == null) return;
        if (MapUtils.isNotEmpty(req.getData())) {
            if (data == null) data = Maps.newHashMap();
            data.putAll(req.getData());
        }
        this.limit = req.getLimit();
    }

    public <T> void put(String name, T value) {
        if (data == null) data = Maps.newHashMap();
        data.put(name, value);
    }

    public <T> T get(String name, T value) {
        if (MapUtils.isEmpty(data)) return value;
        return (T)data.getOrDefault(name, value);
    }

    public <T> T get(String name) {
        return (T)data.getOrDefault(name, null);
    }
}
