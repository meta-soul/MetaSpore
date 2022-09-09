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

package com.dmetasoul.metaspore.recommend.baseservice;

import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.ThreadPoolConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;

import java.util.concurrent.*;
/**
 * 向spring注册线程池
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Component
public class ThreadPoolService {
    public static final int DEFAULT_CORE_POOL_SIZE = 1;
    public static final int DEFAULT_MAX_POOL_SIZE = 10;
    public static final int DEFAULT_KEEP_ALIVE_TIME = 30000;
    public static final TimeUnit DEFAULT_TIME_UNIT = TimeUnit.MILLISECONDS;
    public static final int DEFAULT_CAPACITY = 10000;
    /**
     * 线程池配置信息
     */
    @Autowired
    public ThreadPoolConfig threadPoolConfig;

    /**
     * 向spring注册用于调用数据源source获取数据的线程池
     */
    @Bean
    public ExecutorService sourcePool() {
        ThreadPoolConfig.ThreadPool threadPool = threadPoolConfig.getSource();
        return new ThreadPoolExecutor(
                getCorePoolSize(threadPool),
                getMaximumPoolSize(threadPool),
                getKeepAliveTime(threadPool),
                getTimeunit(threadPool),
                new ArrayBlockingQueue<>(getCapacity(threadPool)), Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }
    /**
     * 向spring注册用于各种计算任务的线程池
     */
    @Bean
    public ExecutorService taskPool() {
        ThreadPoolConfig.ThreadPool threadPool = threadPoolConfig.getTask();
        return new ThreadPoolExecutor(
                getCorePoolSize(threadPool),
                getMaximumPoolSize(threadPool),
                getKeepAliveTime(threadPool),
                getTimeunit(threadPool),
                new ArrayBlockingQueue<>(getCapacity(threadPool)), Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }
    /**
     * 向spring注册用于执行服务工作流程的线程池
     */
    @Bean
    public ExecutorService workFlowPool() {
        ThreadPoolConfig.ThreadPool threadPool = threadPoolConfig.getWorkflow();
        return new ThreadPoolExecutor(
                getCorePoolSize(threadPool),
                getMaximumPoolSize(threadPool),
                getKeepAliveTime(threadPool),
                getTimeunit(threadPool),
                new ArrayBlockingQueue<>(getCapacity(threadPool)), Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }

    private int getCorePoolSize(ThreadPoolConfig.ThreadPool threadPool) {
        if (threadPool == null) return DEFAULT_CORE_POOL_SIZE;
        return Utils.nullThenValue(threadPool.getCorePoolSize(), DEFAULT_CORE_POOL_SIZE);
    }

    private int getMaximumPoolSize(ThreadPoolConfig.ThreadPool threadPool) {
        if (threadPool == null) return DEFAULT_MAX_POOL_SIZE;
        return Utils.nullThenValue(threadPool.getMaximumPoolSize(), DEFAULT_MAX_POOL_SIZE);
    }

    private int getKeepAliveTime(ThreadPoolConfig.ThreadPool threadPool) {
        if (threadPool == null) return DEFAULT_KEEP_ALIVE_TIME;
        return Utils.nullThenValue(threadPool.getKeepAliveTime(), DEFAULT_KEEP_ALIVE_TIME);
    }

    private TimeUnit getTimeunit(ThreadPoolConfig.ThreadPool threadPool) {
        if (threadPool == null) return DEFAULT_TIME_UNIT;
        return Utils.nullThenValue(threadPool.getTimeunit(), DEFAULT_TIME_UNIT);
    }

    private int getCapacity(ThreadPoolConfig.ThreadPool threadPool) {
        if (threadPool == null) return DEFAULT_CAPACITY;
        return Utils.nullThenValue(threadPool.getCapacity(), DEFAULT_CAPACITY);
    }
}
