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

package com.dmetasoul.metaspore.bucketizer;

import com.dmetasoul.metaspore.annotation.BucketizerAnnotation;
import com.dmetasoul.metaspore.configure.ExperimentItem;
import com.dmetasoul.metaspore.data.DataContext;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;

import java.util.List;
import java.util.Map;

@Slf4j
@BucketizerAnnotation("random")
public class RandomLayerBucketizer implements LayerBucketizer {
    protected AliasSampler sampler;
    private List<ExperimentItem> experiments;

    @Override
    public void init(List<ExperimentItem> experiments, Map<String, Object> options) {
        if (CollectionUtils.isEmpty(experiments)) return;
        this.experiments = experiments;
        double[] prob = new double[experiments.size()];
        for (int i = 0; i < experiments.size(); i++) {
            ExperimentItem experimentItem = experiments.get(i);
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