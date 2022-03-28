package com.dmetasoul.metaspore.demo.movielens;

import com.dmetasoul.metaspore.demo.movielens.abtesting.layer.bucketizer.AliasSampler;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class AliasSamplerUnitTest {
    @Test
    public void testAliasSampler() {
        double[] prob = new double[] {0.1, 0.2, 0.3, 0.4};
        int[] cnt = new int[prob.length];
        AliasSampler sampler = new AliasSampler(prob);
        for (int i = 0; i < 10000; i++) {
            cnt[sampler.nextInt()]++;
        }
        for (int i = 0; i < cnt.length; i++) {
            System.out.println(cnt[i]);
        }
    }
}
