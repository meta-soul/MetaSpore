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

package com.dmetasoul.metaspore.recommend.bucketizer;

import com.dmetasoul.metaspore.recommend.annotation.BucketizerAnnotation;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;

import java.util.List;
@BucketizerAnnotation("random")
public class RandomLayerBucketizer implements LayerBucketizer{
    protected AliasSampler sampler;
    private List<RecommendConfig.ExperimentItem> experiments;

    public void init(RecommendConfig.Layer layer) {
        experiments = layer.getExperiments();
        double[] prob = new double[experiments.size()];
        for (int i = 0; i < experiments.size(); i++) {
            RecommendConfig.ExperimentItem experimentItem = experiments.get(i);
            double ratio = experimentItem.getRatio();
            prob[i] = ratio;
        }
        prob = normalize(prob);
        sampler = new AliasSampler(prob);
    }

    @Override
    public String toBucket(DataContext context) {
        int bucketNo = sampler.nextInt() % experiments.size();
        return experiments.get(bucketNo).getName();
    }
}