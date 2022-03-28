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
