package com.dmetasoul.metaspore.demo.movielens.diversify.diversifier;

import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class DiverseProvider {


    private final Map<String, Diversifier> diversifierMap;

    public DiverseProvider(List<Diversifier> diversifiers) {
        this.diversifierMap = new HashMap<>();
        diversifiers.forEach(x -> diversifierMap.put(x.getClass().getSimpleName().toLowerCase(), x));
    }

    public Diversifier getDiversifier(String name) {
        return diversifierMap.get(name.toLowerCase());
    }

    public Collection<Diversifier> getDiversifiers(Collection<String> names) {
        if (names == null) {
            return Collections.emptyList();
        }
        ArrayList<Diversifier> diversifiers = new ArrayList<>();
        for (String name : names) {
            diversifiers.add(getDiversifier(name));
        }
        return diversifiers;
    }
}
