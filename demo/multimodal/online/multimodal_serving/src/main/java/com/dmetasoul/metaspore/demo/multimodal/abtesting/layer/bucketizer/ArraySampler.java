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

package com.dmetasoul.metaspore.demo.multimodal.abtesting.layer.bucketizer;

import com.google.common.hash.HashCode;

public class ArraySampler {
    private int[] searchArray;

    public ArraySampler(double[] prob) {
        this(prob, 1000);
    }

    public ArraySampler(double[] prob, int precision) {
        this.searchArray = new int[precision];
        int counter = 0;
        for (int i = 0; i < prob.length; i++) {
            for (int j = 0; j < prob[i] * precision && counter < precision; j++, counter++) {
                searchArray[counter] = i;
            }
        }
        while (counter < precision) {
            searchArray[counter] = searchArray.length - 1;
        }
    }

    public int nextInt(HashCode hashCode) {
        long hashValue = getUnsignedInt(hashCode.asInt());
        return searchArray[(int) (hashValue % searchArray.length)];
    }

    public static long getUnsignedInt(int x) {
        return x & 0x00000000ffffffffL;
    }
}