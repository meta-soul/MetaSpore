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

package com.dmetasoul.metaspore.common;

import com.google.common.collect.Maps;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.config.YamlPropertiesFactoryBean;
import org.springframework.core.env.EnumerablePropertySource;
import org.springframework.core.io.ByteArrayResource;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * 用于spring bean读取本地yaml格式配置文件
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
public class ServicePropertySource<T> extends EnumerablePropertySource<T> {
    private final Map<String, Object> properties = Maps.newLinkedHashMap();
    public ServicePropertySource(String name, T source) {
        super(name, source);
    }

    public void updateConfigByString(String content, Format format) {
        if (StringUtils.isEmpty(content)) {
            return;
        }
        Properties prop = generateProperties(content, format);
        prop.forEach((key, value)->{
            this.properties.put(String.valueOf(key), value);
        });
    }

    protected Properties generateProperties(String value, Format format) {
        Properties props = new Properties();
        if (format == Format.PROPERTIES) {
            try {
                props.load(new ByteArrayInputStream(value.getBytes(StandardCharsets.ISO_8859_1)));
                return props;
            } catch (IOException var5) {
                throw new IllegalArgumentException(value + " can't be encoded using ISO-8859-1");
            }
        } else if (format == Format.YAML) {
            YamlPropertiesFactoryBean yaml = new YamlPropertiesFactoryBean();
            yaml.setResources(new ByteArrayResource(value.getBytes(StandardCharsets.UTF_8)));
            return yaml.getObject();
        } else {
            return props;
        }
    }

    @Override
    @SuppressWarnings("NullableProblems")
    public String[] getPropertyNames() {
        Set<String> strings = this.properties.keySet();
        return strings.toArray(new String[0]);
    }

    @Override
    public Object getProperty(@NonNull String name) {
        return this.properties.get(name);
    }

    public static enum Format {
        PROPERTIES,
        YAML;
    }
}
