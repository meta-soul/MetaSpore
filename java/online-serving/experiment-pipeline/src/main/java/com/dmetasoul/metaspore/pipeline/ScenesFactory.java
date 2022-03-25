package com.dmetasoul.metaspore.pipeline;

import com.dmetasoul.metaspore.pipeline.pojo.SceneConfig;

import java.util.Map;

public interface ScenesFactory {
    Scene getScene(String sceneName);

    SceneConfig getSceneConfig();

    Map<String, Scene> getScenes();
}
