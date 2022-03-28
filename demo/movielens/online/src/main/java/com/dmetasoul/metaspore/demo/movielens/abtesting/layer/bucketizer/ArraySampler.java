package com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer;

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
