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
import org.springframework.util.Assert;

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
    private List<Scene> scenes;
    private List<Service> services;

    /**
     * Service 直接调用一个或多个algoTransform task获取结果 或者 接受一个或多个DataResult结果， 然后进行transform计算
     * 接受到一个或多个DataResult结果时，
     * 对于多个DataResult，要求 featureTable schema 相同或者包含相同的columnNames type， merge时根据option配置决定去重
     * 接受到的DataResult merge后 缓存到context中， 参与后续的task计算
     */
    @Data
    public static class Service {
        private String name;
        private String taskName;
        private List<String> tasks;
        private Map<String, Object> options;
        private List<String> columnNames;
        private Map<String, String> columnMap;
        private List<Map<String, String>> columns;
        private List<TransformConfig> preTransforms;
        private List<TransformConfig> transforms;

        public void setTasks(List<String> tasks) {
            if (CollectionUtils.isEmpty(tasks)) return;
            this.tasks = tasks;
        }

        public void setTasks(String task) {
            if (StringUtils.isEmpty(task)) return;
            this.tasks = List.of(task);
        }
        public void setColumns(List<Map<String, String>> columns) {
            if (CollectionUtils.isNotEmpty(columns)) {
                this.columnNames = Lists.newArrayList();
                this.columnMap = Maps.newHashMap();
                columns.forEach(map -> map.forEach((x, y) -> {
                    columnNames.add(x);
                    this.columnMap.put(x, y);
                }));
                this.columns = columns;
            }
        }
        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("Service config name must not be empty!");
                return false;
            }
            if (StringUtils.isEmpty(taskName)) {
                taskName = "Service";
            }
            return true;
        }
    }
    /**
     * Experiment 执行Dag任务，每个任务执行完毕后将结果传递给下游任务参与计算
     * Dag执行过程与DataService不相同
     * 不变更data schema
     */
    @Data
    public static class Experiment {
        private String name;
        private String taskName;
        private List<Chain> chains;
        private Map<String, Object> options;

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("Experiment config name must not be empty!");
                return false;
            }
            if (CollectionUtils.isEmpty(chains)) {
                log.error("Experiment config chains must not be empty!");
                return false;
            }
            for (Chain chain : chains) {
                if (!chain.checkAndDefault()) {
                    log.error("Experiment config chain must be right!");
                    return false;
                }
            }
            if (StringUtils.isEmpty(taskName)) {
                taskName = "Experiment";
            }
            return true;
        }
    }

    /**
     *  执行bucketizer 分桶 然后计算
     *  不改变data schema
     */
    @Data
    public static class ExperimentItem {
        private String name;
        private double ratio;
    }
    @Data
    public static class Layer {
        private String name;
        private String taskName;
        private List<ExperimentItem> experiments;
        private String bucketizer;
        private Map<String, Object> options;
        private double sumRatio = 0.0;

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
            if (StringUtils.isEmpty(taskName)) {
                taskName = "Layer";
            }
            return true;
        }
    }

    /**
     * 执行layer组成的任务Dag， 任务上下游依赖
     */
    @Data
    public static class Scene {
        private String name;
        private String taskName;
        private List<Chain> chains;

        private Map<String, Object> options;
        private List<String> columnNames;
        private Map<String, String> columnMap;
        private List<Map<String, String>> columns;

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
            if (StringUtils.isEmpty(taskName)) {
                taskName = "Scene";
            }
            Assert.notNull(columnNames, "scene must configure output columns! at " + name);
            return true;
        }
        public void setColumns(List<Map<String, String>> columns) {
            if (CollectionUtils.isNotEmpty(columns)) {
                this.columnNames = Lists.newArrayList();
                this.columnMap = Maps.newHashMap();
                columns.forEach(map -> map.forEach((x, y) -> {
                    columnNames.add(x);
                    this.columnMap.put(x, y);
                }));
                this.columns = columns;
            }
        }
    }
}
