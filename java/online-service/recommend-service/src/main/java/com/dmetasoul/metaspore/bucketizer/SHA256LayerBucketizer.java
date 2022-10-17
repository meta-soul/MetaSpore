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
import com.google.common.base.Charsets;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import org.apache.commons.collections4.CollectionUtils;

import java.util.List;
import java.util.Map;

// References:
// * https://mojito.mx/docs/example-hash-function-split-test-assignment
// * https://engineering.depop.com/a-b-test-bucketing-using-hashing-475c4ce5d07
@SuppressWarnings("UnstableApiUsage")
@BucketizerAnnotation("sha256")
public class SHA256LayerBucketizer implements LayerBucketizer {
    protected ArraySampler sampler;

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
        System.out.println("RandomLayer layer, init sampler...");
        prob = normalize(prob);
        sampler = new ArraySampler(prob);
    }

    @Override
    public String toBucket(DataContext context) {
        HashCode sha256 = sha256(context.getId());
        int bucketNo = sampler.nextInt(sha256) % experiments.size();
        return experiments.get(bucketNo).getName();
    }

    protected HashCode sha256(String id) {
        Hasher hasher = Hashing.sha256().newHasher();
        hasher.putString(id, Charsets.UTF_8);
        return hasher.hash();
    }
}