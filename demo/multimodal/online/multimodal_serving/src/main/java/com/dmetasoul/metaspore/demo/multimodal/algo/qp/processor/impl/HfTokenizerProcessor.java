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

package com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor.impl;

import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor.Processor;
import com.dmetasoul.metaspore.demo.multimodal.model.SearchContext;
import com.dmetasoul.metaspore.demo.multimodal.service.HfPreprocessorService;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.google.protobuf.ByteString;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Service
public class HfTokenizerProcessor implements Processor {
    private final HfPreprocessorService hfPreprocessorService;

    public HfTokenizerProcessor(HfPreprocessorService hfPreprocessorService) {
        this.hfPreprocessorService = hfPreprocessorService;
    }

    @Override
    public Map<String, ByteString> process(SearchContext searchContext, QueryModel queryModel) throws IOException {
        String modelName = searchContext.getQpQueryEmbeddingModelName();
        List<String> texts = List.of(queryModel.getQuery());
        Map<String, ByteString> serviceResults = hfPreprocessorService.predictBlocking(modelName, texts, Collections.emptyMap());

        // There are some bugs in arrowTensor encode-decode
        //System.out.println("Qp Processor Results:");
        //System.out.println(hfPreprocessorService.getIntPredictFromArrowTensorResult(hfPreprocessorService.pbToArrow(serviceResults)));

        return serviceResults;
    }
}
