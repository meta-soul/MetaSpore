package com.dmetasoul.metaspore.pipeline.pojo;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.List;

@RefreshScope
@ConfigurationProperties(prefix = "scene-config")
@Data
@Component
public class SceneConfig {
    private List<Scene> scenes;
    private List<Experiment> experiments;

    @PostConstruct
    public void postConstruct() {
    }
}
