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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Random;

// Reference:
// * https://blog.csdn.net/mandycool/article/details/8182672

public class AliasSampler {
    private double[] probability;

    private int[] alias;

    private int length;

    private Random rand;

    public AliasSampler(double[] prob) {
        this(prob, new Random());
    }

    public AliasSampler(double[] prob, Random rand) {
        if (prob == null || rand == null) {
            throw new NullPointerException();
        }

        if (prob.length == 0) {
            throw new IllegalArgumentException();
        }

        this.rand = rand;
        this.length = prob.length;
        this.probability = new double[length];
        this.alias = new int[length];

        double[] probTemp = new double[length];
        Deque<Integer> small = new ArrayDeque<>();
        Deque<Integer> large = new ArrayDeque<>();

        for (int i = 0; i < length; i++) {
            probTemp[i] = prob[i] * length;
            if (probTemp[i] < 1.0) {
                small.add(i);
            } else {
                large.add(i);
            }
        }

        while (!small.isEmpty() && !large.isEmpty()) {
            int less = small.pop();
            int more = large.pop();
            probability[less] = probTemp[less];
            alias[less] = more;
            probTemp[more] = probTemp[more] - (1.0 - probability[less]);
            if (probTemp[more] < 1.0) {
                small.add(more);
            } else {
                large.add(more);
            }
        }

        while (!small.isEmpty()) {
            probability[small.pop()] = 1.0;
        }

        while (!large.isEmpty()) {
            probability[large.pop()] = 1.0;
        }
    }

    public int nextInt() {
        int column = rand.nextInt(length);
        boolean coinToss = rand.nextDouble() < probability[column];
        return coinToss ? column : alias[column];
    }
}