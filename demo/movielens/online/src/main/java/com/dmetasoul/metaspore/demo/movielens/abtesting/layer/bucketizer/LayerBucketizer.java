package com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer;

import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;

public interface LayerBucketizer {
    public String toBucket(RecommendContext context);

    public default double[] normalize(double[] prob) {
        double total = 0;
        for (int i = 0; i < prob.length; i++) {
            total += prob[i];
        }
        if (total == 0) {
            throw new IllegalArgumentException("Sum of probability is zero...");
        }

        double[] probability = new double[prob.length];
        for (int i = 0; i < probability.length; i++) {
            probability[i] = prob[i] / total;
        }

        return probability;
    }
}
