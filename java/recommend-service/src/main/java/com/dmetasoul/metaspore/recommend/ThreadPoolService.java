package com.dmetasoul.metaspore.recommend;

import com.dmetasoul.metaspore.recommend.configure.ThreadPoolConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;

import java.util.concurrent.*;

@Component
public class ThreadPoolService {
    @Autowired
    public ThreadPoolConfig threadPoolConfig;

    @Bean
    public ExecutorService featurePool() {
        return new ThreadPoolExecutor(
                threadPoolConfig.getFeature().getCorePoolSize(),
                threadPoolConfig.getFeature().getMaximumPoolSize(),
                threadPoolConfig.getFeature().getKeepAliveTime(),
                threadPoolConfig.getFeature().getTimeunit(),
                new ArrayBlockingQueue<Runnable>(threadPoolConfig.getFeature().getCapacity()), Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }

    @Bean
    public ExecutorService taskPool() {
        return new ThreadPoolExecutor(
                threadPoolConfig.getTask().getCorePoolSize(),
                threadPoolConfig.getTask().getMaximumPoolSize(),
                threadPoolConfig.getTask().getKeepAliveTime(),
                threadPoolConfig.getTask().getTimeunit(),
                new ArrayBlockingQueue<Runnable>(threadPoolConfig.getTask().getCapacity()), Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }

    @Bean
    public ExecutorService workFlowPool() {
        return new ThreadPoolExecutor(
                threadPoolConfig.getWorkflow().getCorePoolSize(),
                threadPoolConfig.getWorkflow().getMaximumPoolSize(),
                threadPoolConfig.getWorkflow().getKeepAliveTime(),
                threadPoolConfig.getWorkflow().getTimeunit(),
                new ArrayBlockingQueue<Runnable>(threadPoolConfig.getWorkflow().getCapacity()), Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }
}
