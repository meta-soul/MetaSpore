package com.dmetasoul.metaspore.demo.movielens.controller;

import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.model.request.PayloadParam;
import com.dmetasoul.metaspore.pipeline.ScenesFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;

@RestController
public class RecommendDebugController {
    @Autowired
    private ScenesFactory scenes;

    @PostMapping(value="/user/{userId}/debug")
    public RecommendResult recommend(@PathVariable("userId") String userId,
                                     @RequestBody PayloadParam payloadParam) throws IOException {
        System.out.println("MovieLens recommendation, scene getScenesConfig: " + scenes.getSceneConfig());
        System.out.println("MovieLens recommendation, request payload parameter: " + payloadParam);
        if (payloadParam != null && payloadParam.useDebug) {
            return (RecommendResult) scenes.getScene("guess-you-like").runDebug(
                    new RecommendContext(userId, payloadParam), payloadParam.specifiedLayerAndExperiment);
        } else {
            return (RecommendResult) scenes.getScene("guess-you-like").run(new RecommendContext(userId));
        }
    }

}
