package com.dmetasoul.metaspore.demo.movielens.controller;

import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.service.RecommendService;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
@RequestMapping("/verify")
public class RecommendVerifyController {
    private final RecommendService recommendService;

    public RecommendVerifyController(RecommendService recommendService) {
        this.recommendService = recommendService;
    }

    @RequestMapping(value="/user/{userId}")
    public RecommendResult recommend(@PathVariable("userId") String userId) throws IOException {
        return recommendService.recommend(new RecommendContext(userId));
    }
}
