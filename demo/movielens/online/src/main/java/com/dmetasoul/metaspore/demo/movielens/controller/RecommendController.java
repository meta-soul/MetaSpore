//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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