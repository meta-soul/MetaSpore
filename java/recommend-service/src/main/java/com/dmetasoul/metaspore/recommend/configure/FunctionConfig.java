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
package com.dmetasoul.metaspore.recommend.configure;

import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.context.annotation.Configuration;

import java.util.*;

/**
 * UDF函数相关配置类
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
@Configuration
@RefreshScope
@ConfigurationProperties(prefix = "function-config")
public class FunctionConfig {
    private String path;
    private List<JarInfo> jars;
    @Data
    public static class JarInfo {
        private String name;
        private List<Map<String, String>> fieldFunction;
        private List<Map<String, String>> transformFunction;
        private List<Map<String, String>> transformMergeOperator;
        private List<Map<String, String>> transformUpdateOperator;
        private List<Map<String, String>> layerBucketizer;
        private List<Map<String, String>> sourceTableTask;
        private List<Map<String, String>> algoTransformTask;
        private List<Map<String, String>> recommendService;
        private List<Map<String, String>> dataSource;
    }
}
