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

import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.util.set.Sets;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.context.annotation.Configuration;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
/**
 * 推荐服务配置类
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
@Configuration
@RefreshScope
@ConfigurationProperties(prefix = "recommend-service")
public class RecommendConfig {
    private List<Layer> layers;
    private List<Experiment> experiments;
    private List<Service> services;
    private List<Scene> scenes;

    @Data
    public static class Service {
        private String name;
        private List<String> depend;
        private Map<String, Object> options;
        private String serviceName;

        private List<String> columnNames;
        private Map<String, String> columnMap;
        private List<Map<String, String>> columns;

        public void setDepend(List<String> list) {
            depend = list;
        }

        public void setDepend(String str) {
            depend = List.of(str);
        }

        public void setColumnMap(List<Map<String, String>> columnMap) {
            if (CollectionUtils.isNotEmpty(columnMap)) {
                this.columnNames = Lists.newArrayList();
                this.columnMap = Maps.newHashMap();
                columnMap.forEach(map -> map.forEach((x, y) -> {
                    columnNames.add(x);
                    this.columnMap.put(x, y);
                }));
            }
        }

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("Service config name must not be empty!");
                return false;
            }
            setColumnMap(columns);
            if (CollectionUtils.isEmpty(columnNames)) {
                log.error("Service config must be output columns!");
                return false;
            }
            for (Map.Entry<String, String> entry : columnMap.entrySet()) {
                if (!DataTypes.typeIsSupport(entry.getValue())) {
                    log.error("Service Output columns config columns type:{} must be support!", entry.getValue());
                    return false;
                }
            }
            if (StringUtils.isEmpty(serviceName)) {
                serviceName = name;
            }
            return true;
        }
    }

    @Data
    public static class Chain {
        private String name;
        private List<String> then;
        private List<String> when;

        private boolean isAny;

        private Long timeOut;

        private TimeUnit timeOutUnit;
        private List<String> columnNames;
        private Map<String, String> columnMap;
        private List<Map<String, String>> columns;

        public void setThen(List<String> list) {
            then = list;
        }

        public void setThen(String str) {
            then = List.of(str);
        }

        public void setWhen(List<String> list) {
            when = list;
        }

        public void setWhen(String str) {
            when = List.of(str);
        }

        public Chain() {}

        public Chain(List<String> then, List<String> when, boolean isAny, Long timeOut, TimeUnit timeOutUnit) {
            this.then = then;
            this.when = when;
            this.isAny = isAny;
            this.timeOut = timeOut;
            this.timeOutUnit = timeOutUnit;
        }

        public void setColumnMap(List<Map<String, String>> columnMap) {
            if (CollectionUtils.isNotEmpty(columnMap)) {
                this.columnNames = Lists.newArrayList();
                this.columnMap = Maps.newHashMap();
                columnMap.forEach(map -> map.forEach((x, y) -> {
                    columnNames.add(x);
                    this.columnMap.put(x, y);
                }));
                if (CollectionUtils.isEmpty(columns)) {
                    this.columns = columnMap;
                }
            }
        }

        public void setTimeOutUnit(String data) {
            switch (Strings.capitalize(data.toLowerCase())) {
                case "Nanos":
                    timeOutUnit = TimeUnit.NANOSECONDS;
                    break;
                case "Micros":
                    timeOutUnit = TimeUnit.MICROSECONDS;
                    break;
                case "Millis":
                    timeOutUnit = TimeUnit.MILLISECONDS;
                    break;
                case "Seconds":
                    timeOutUnit = TimeUnit.SECONDS;
                    break;
                case "Minutes":
                    timeOutUnit = TimeUnit.MINUTES;
                    break;
                case "Hours":
                    timeOutUnit = TimeUnit.HOURS;
                    break;
                case "Days":
                    timeOutUnit = TimeUnit.DAYS;
                    break;
                default:
                    log.warn("threadpool keepAliveTime timeunit default is seconds , config is {}", data);
                    timeOutUnit = TimeUnit.SECONDS;
            }
        }

        public boolean checkAndDefault() {
            if (CollectionUtils.isEmpty(then) && CollectionUtils.isEmpty(when)) {
                log.error("the chain must has node in then or when!");
                return false;
            }
            setColumnMap(columns);
            if (CollectionUtils.isNotEmpty(when) && CollectionUtils.isEmpty(columnNames)) {
                log.error("the chain:{} must has columns while has when!", this);
                return false;
            }
            if (CollectionUtils.isNotEmpty(when)) {
                for (Map.Entry<String, String> entry : columnMap.entrySet()) {
                    if (!DataTypes.typeIsSupport(entry.getValue())) {
                        log.error("Output columns config columns type:{} must be support!", entry.getValue());
                        return false;
                    }
                }
            }
            Set<String> taskSet = Sets.newHashSet();
            int taskNum = 0;
            if (CollectionUtils.isNotEmpty(then)) {
                taskSet.addAll(then);
                taskNum += then.size();
            }
            if (CollectionUtils.isNotEmpty(when)) {
                taskSet.addAll(when);
                taskNum += when.size();
            }
            if (taskNum != taskSet.size()) {
                log.error("the chain has duplicate task in then or when!");
                return false;
            }
            setDefaultTimeOut(30000L, TimeUnit.MILLISECONDS);
            return true;
        }

        public void setDefaultTimeOut(long time, TimeUnit unit) {
            if (timeOut == null) {
                timeOut = time;
            }
            if (timeOutUnit == null) {
                timeOutUnit = unit;
            }
        }
    }
    @Data
    public static class Experiment {
        private String name;
        private List<Chain> chains;
        private Map<String, Object> options;
        private List<String> columnNames;
        private Map<String, String> columnMap;

        private List<Map<String, String>> columns;

        public void setColumnMap(List<Map<String, String>> columnMap) {
            if (CollectionUtils.isNotEmpty(columnMap)) {
                this.columnNames = Lists.newArrayList();
                this.columnMap = Maps.newHashMap();
                columnMap.forEach(map -> map.forEach((x, y) -> {
                    columnNames.add(x);
                    this.columnMap.put(x, y);
                }));
                if (CollectionUtils.isEmpty(columns)) {
                    this.columns = columnMap;
                }
            }
        }
        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("Experiment config name must not be empty!");
                return false;
            }
            if (CollectionUtils.isEmpty(chains)) {
                log.error("Experiment config chains must not be empty!");
                return false;
            }
            for (int index = 0; index < chains.size(); ++index) {
                RecommendConfig.Chain chain = chains.get(index);
                if (!chain.checkAndDefault()) {
                    log.error("Experiment config chain must be right!");
                    return false;
                }
            }
            if (MapUtils.isNotEmpty(options)) {
                if (options.containsKey("maxReservation") && !options.containsKey("cutField")) {
                    log.error("Experiment config maxReservation must be config cutField!");
                    return false;
                }
            }
            return true;
        }
    }

    @Data
    public static class ExperimentItem {
        private String name;
        private double ratio;
    }
    @Data
    public static class Layer {
        private String name;
        private List<ExperimentItem> experiments;
        private String bucketizer;
        private Map<String, Object> options;
        private double sumRatio = 0.0;

        private List<String> columnNames;
        private Map<String, String> columnMap;

        private List<Map<String, String>> columns;

        public void setColumnMap(List<Map<String, String>> columnMap) {
            if (CollectionUtils.isNotEmpty(columnMap)) {
                this.columnNames = Lists.newArrayList();
                this.columnMap = Maps.newHashMap();
                columnMap.forEach(map -> map.forEach((x, y) -> {
                    columnNames.add(x);
                    this.columnMap.put(x, y);
                }));
                if (CollectionUtils.isEmpty(columns)) {
                    this.columns = columnMap;
                }
            }
        }

        public void setExperiments(List<ExperimentItem> list) {
            list.forEach(x-> {
                sumRatio += x.ratio;
            });
            experiments = list;
        }

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("Layer config name must not be empty!");
                return false;
            }
            if (StringUtils.isEmpty(bucketizer)) {
                bucketizer = "random";
            }
            if (CollectionUtils.isEmpty(experiments) || Sets.newHashSet(experiments).size() != experiments.size()) {
                log.error("Layer config experiments must not be empty or has duplicate experiment!");
                return false;
            }
            if (Math.abs(sumRatio - 0.0) < 1e-6) {
                log.error("Layer experiments ratio sum must not be 0.0!");
                return false;
            }
            return true;
        }
    }

    @Data
    public static class Scene {
        private String name;
        private List<Chain> chains;

        private Map<String, Object> options;
        private List<String> columnNames;
        private Map<String, String> columnMap;

        private List<Map<String, String>> columns;

        public void setColumnMap(List<Map<String, String>> columnMap) {
            if (CollectionUtils.isNotEmpty(columnMap)) {
                this.columnNames = Lists.newArrayList();
                this.columnMap = Maps.newHashMap();
                columnMap.forEach(map -> map.forEach((x, y) -> {
                    columnNames.add(x);
                    this.columnMap.put(x, y);
                }));
                if (CollectionUtils.isEmpty(columns)) {
                    this.columns = columnMap;
                }
            }
        }

        private Map<String, Chain> chainMap = Maps.newHashMap();

        public void setChains(List<Chain> data) {
            if (CollectionUtils.isEmpty(data)) return;
            data.forEach(chain -> {
                if (StringUtils.isNotEmpty(chain.getName())) {
                    chainMap.put(chain.getName(), chain);
                }
            });
            chains = data;
        }

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name) || CollectionUtils.isEmpty(chains)) {
                log.error("Scene config name and chains must not be empty!");
                return false;
            }
            for (Chain chain : chains) {
                if (!chain.checkAndDefault()) {
                    log.error("Scene config chain must be right!");
                    return false;
                }
            }
            return true;
        }
    }
}
