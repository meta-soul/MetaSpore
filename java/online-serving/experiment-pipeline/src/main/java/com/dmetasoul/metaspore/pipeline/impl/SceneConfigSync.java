package com.dmetasoul.metaspore.pipeline.impl;

import com.dmetasoul.metaspore.pipeline.pojo.SceneConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;


@Component
public class SceneConfigSync {

    @Autowired
    private SceneConfig sceneConfig;

    @Bean
    @RefreshScope
    public ScenesFactoryImpl RefreshScenes(ApplicationContext ctx) {
        return new ScenesFactoryImpl(sceneConfig, ctx);
    }

}
