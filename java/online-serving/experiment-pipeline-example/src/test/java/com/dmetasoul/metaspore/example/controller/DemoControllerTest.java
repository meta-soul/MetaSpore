package com.dmetasoul.metaspore.example.controller;

import com.dmetasoul.metaspore.example.layer.MyExperimentPojo;
import com.dmetasoul.metaspore.example.layer.MyExperimentPojo2;
import com.dmetasoul.metaspore.pipeline.Scene;
import com.dmetasoul.metaspore.pipeline.ScenesFactory;
import com.dmetasoul.metaspore.pipeline.pojo.SceneConfig;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;
import org.springframework.util.Assert;

import java.util.HashMap;
import java.util.Map;

@SpringBootTest
class DemoControllerTest {
    @Autowired
    private ScenesFactory scenes;

    @Test
    void getSceneConfig() {
        SceneConfig sceneConfig = scenes.getSceneConfig();
        System.out.println("scene getScene: " + sceneConfig);
        Assert.notNull(sceneConfig, "sceneConfig is null");
    }

    @Test
    void getScenes() {
        Map<String, Scene> scenes = this.scenes.getScenes();
        System.out.println("scene getScene: " + scenes);
        Assert.notNull(scenes, "scenes is null");

    }

    @Test
    void recommand() {
        MyExperimentPojo input = new MyExperimentPojo();
        input.setUserId("1");
        MyExperimentPojo2 result = (MyExperimentPojo2) scenes.getScene("guess-you-like").run(input);
        System.out.println(result);
    }

    @Test
    void debug() {
        HashMap<String, String> specifiedLayerAndExperiment = new HashMap<>();
        specifiedLayerAndExperiment.put("recall", "milvus2");
        MyExperimentPojo input = new MyExperimentPojo();
        input.setUserId("1");
        MyExperimentPojo2 result = (MyExperimentPojo2) scenes.getScene("guess-you-like").runDebug(input, specifiedLayerAndExperiment);
        System.out.println(result);

    }
}