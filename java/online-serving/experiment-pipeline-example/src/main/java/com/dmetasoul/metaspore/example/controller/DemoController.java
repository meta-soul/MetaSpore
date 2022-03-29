package com.dmetasoul.metaspore.example.controller;


import com.dmetasoul.metaspore.example.layer.FirstLayerPojo;
import com.dmetasoul.metaspore.example.layer.SecondLayerPojo;
import com.dmetasoul.metaspore.pipeline.ScenesFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/scene")
public class DemoController {

    @Autowired
    private ScenesFactory scenes;

    @RequestMapping("/scenes")
    public void getScenes() {
        System.out.println("scene getScene: " + scenes.getScenes());
    }

    @RequestMapping("/sceneConfig")
    public void getSceneConfig() {
        System.out.println("scene getScenesConfig: " + scenes.getSceneConfig());
    }

}
