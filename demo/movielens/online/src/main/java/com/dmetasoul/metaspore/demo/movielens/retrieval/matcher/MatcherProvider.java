package com.dmetasoul.metaspore.demo.movielens.retrieval.matcher;

import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class MatcherProvider {

    private final Map<String, Matcher> matcherMap;

    public MatcherProvider(List<Matcher> matchers) {
        this.matcherMap = new HashMap<>();
        matchers.forEach(x -> matcherMap.put(x.getClass().getSimpleName(), x));
    }

    public Matcher getMatcher(String name) {
        return matcherMap.get(name);
    }

    public Collection<Matcher> getMatchers(Collection<String> names) {
        if (names == null) {
            return Collections.emptyList();
        }

        ArrayList<Matcher> matchers = new ArrayList<>();
        for (String name : names) {
            matchers.add(getMatcher(name));
        }

        return matchers;
    }
}
