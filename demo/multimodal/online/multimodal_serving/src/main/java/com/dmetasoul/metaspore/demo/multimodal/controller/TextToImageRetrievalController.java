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

package com.dmetasoul.metaspore.demo.multimodal.controller;

import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchResult;
import com.dmetasoul.metaspore.demo.multimodal.model.request.PayloadParam;
import com.dmetasoul.metaspore.pipeline.ScenesFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
public class TextToImageRetrievalController {
    @Autowired
    private ScenesFactory scenes;

    @PostMapping(value="t2i/user/{userId}")
    public SearchResult search(@PathVariable("userId") String userId,
                               @RequestBody PayloadParam payloadParam) throws IOException {
        System.out.println("Multi-Modal Retrieval, scene getScenesConfig: " + scenes.getSceneConfig());
        System.out.println("Multi-Modal Retrieval, userId:" + userId + ", request payload parameter:" + payloadParam);

        if (payloadParam != null && payloadParam.useDebug) {
            return (SearchResult) scenes.getScene("textToImage").runDebug(
                    new SearchResult(payloadParam.query, new SearchContext(userId)),
                    payloadParam.specifiedLayerAndExperiment);
        } else {
            return (SearchResult) scenes.getScene("textToImage").run(
                    new SearchResult(payloadParam.query, new SearchContext(userId)));
        }
    }
}
