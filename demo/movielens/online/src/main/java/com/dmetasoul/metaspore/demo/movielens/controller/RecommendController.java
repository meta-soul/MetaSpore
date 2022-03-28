package com.dmetasoul.metaspore.demo.movielens.controller;

import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.pipeline.ScenesFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
public class RecommendController {
    @Autowired
    private ScenesFactory scenes;

    @RequestMapping(value="/user/{userId}")
    public RecommendResult recommend(@PathVariable("userId") String userId) throws IOException {
        System.out.println("MovieLens recommendation, scene getScenesConfig: " + scenes.getSceneConfig());
        RecommendResult result = (RecommendResult) scenes.getScene("guess-you-like").run(new RecommendContext(userId));
        return result;
    }

}
