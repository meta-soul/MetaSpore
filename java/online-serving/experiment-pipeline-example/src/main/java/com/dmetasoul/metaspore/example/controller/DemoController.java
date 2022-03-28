package com.dmetasoul.metaspore.example.controller;


import com.dmetasoul.metaspore.example.layer.MyExperimentPojo;
import com.dmetasoul.metaspore.example.layer.MyExperimentPojo2;
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

    @GetMapping("/recommend")
    public MyExperimentPojo2 recommand(@RequestParam String userId) {
        MyExperimentPojo input = new MyExperimentPojo();
        input.setUserId(userId);
        MyExperimentPojo2 result = (MyExperimentPojo2) scenes.getScene("guess-you-like").run(input);
        return result;
    }
}
