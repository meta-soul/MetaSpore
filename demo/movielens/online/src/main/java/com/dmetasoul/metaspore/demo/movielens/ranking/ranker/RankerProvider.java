package com.dmetasoul.metaspore.demo.movielens.ranking.ranker;

import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class RankerProvider {

    private final Map<String, Ranker> rankerMap;

    public RankerProvider(List<Ranker> rankers) {
        this.rankerMap = new HashMap<>();
        rankers.forEach(x -> rankerMap.put(x.getClass().getSimpleName(), x));
    }

    public Ranker getRanker(String name) {
        return rankerMap.get(name);
    }

    public Collection<Ranker> getRankers(Collection<String> names) {
        if (names == null) {
            return Collections.emptyList();
        }

        ArrayList<Ranker> rankers = new ArrayList<>();
        for (String name : names) {
            rankers.add(getRanker(name));
        }

        return rankers;
    }
}
