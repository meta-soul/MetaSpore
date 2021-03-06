package com.dmetasoul.metaspore.example.controller;

import com.dmetasoul.metaspore.example.layer.FirstLayerPojo;
import com.dmetasoul.metaspore.example.layer.SecondLayerPojo;
import com.dmetasoul.metaspore.pipeline.Scene;
import com.dmetasoul.metaspore.pipeline.ScenesFactory;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import com.dmetasoul.metaspore.pipeline.pojo.SceneConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.util.Assert;

import java.util.HashMap;
import java.util.Map;

@SpringBootTest
class DemoControllerTest {
    @Autowired
    private ScenesFactory scenes;

    private FirstLayerPojo input;

    private Context ctx;

    private HashMap<String, String> specifiedLayerAndExperiment = new HashMap<>();

    @BeforeEach
    void setUp() {
        input = new FirstLayerPojo("1");
        ctx = new Context();
        ctx.setCustomData("my-data");
        specifiedLayerAndExperiment.put("recall", "RecallExperimentTwo");

    }

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
        SecondLayerPojo result = (SecondLayerPojo) scenes.getScene("guess-you-like").run(input);
        System.out.println(result);
    }

    @Test
    void recommandWithContext() {
        SecondLayerPojo result = (SecondLayerPojo) scenes.getScene("guess-you-like").run(input, ctx);
        System.out.println(result);
        System.out.println("ctx.getCustomData(): " + ctx.getCustomData());
    }


    @Test
    void debug() {
        SecondLayerPojo result = (SecondLayerPojo) scenes.getScene("guess-you-like").runDebug(input, specifiedLayerAndExperiment);
        System.out.println(result);
    }

    @Test
    void debugWithContext() {
        SecondLayerPojo result = (SecondLayerPojo) scenes.getScene("guess-you-like").runDebug(input, ctx, specifiedLayerAndExperiment);
        System.out.println(result);
        System.out.println("ctx.getCustomData(): " + ctx.getCustomData());
    }
}