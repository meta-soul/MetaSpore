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

package com.dmetasoul.metaspore.demo.movielens;

import com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer.ArraySampler;
import com.google.common.hash.HashCode;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class ArraySamplerUnitTest {
    @Test
    public void testArraySampler() {
        double[] prob = new double[] {0.3, 0.1, 0.4, 0.2};
        int[] cnt = new int[prob.length];
        ArraySampler sampler = new ArraySampler(prob);
        for (int i = 0; i < 10000; i++) {
            cnt[sampler.nextInt(HashCode.fromInt(i))]++;
        }
        for (int i = 0; i < cnt.length; i++) {
            System.out.println(cnt[i]);
        }
    }
}