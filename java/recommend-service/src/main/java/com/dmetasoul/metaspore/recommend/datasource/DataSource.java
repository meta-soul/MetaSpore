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

import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.concurrent.ExecutorService;

/**
 * DataSource的base类， 用于初始化连接各种数据源
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@Slf4j
@Data
public abstract class DataSource {
    /**
     *  datasource 名称 与配置feature-config中的source.name对应
     */
    protected String name;
    /**
     *  datasource 用来连接获取数据源中数据的线程池
     */
    protected ExecutorService featurePool;
    /**
     *  配置数据对象
     */
    protected TaskFlowConfig taskFlowConfig;

    /**
     *  datasource base 类初始化， 外部使用datasource需要调用此函数进行初始化
     */
    public boolean init(String name, TaskFlowConfig taskFlowConfig, ExecutorService featurePool) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null, init fail!");
            return false;
        }
        this.name = name;
        this.featurePool = featurePool;
        this.taskFlowConfig = taskFlowConfig;
        return initService();
    }
    /**
     *  datasource 具体实现子类初始化， 由base类init函数调用，对外不可见
     */
    protected abstract boolean initService();
    /**
     *  用于datasource 具体实现子类关闭数据源连接等操作
     */
    public abstract void close();
    /**
     *  datasource 根据request和context获取数据，目前除source=request外，获取数据的操作均转移到具体的SourceTableTask中实现
     */
    public DataResult process(ServiceRequest request, DataContext context) {return null;}
}
