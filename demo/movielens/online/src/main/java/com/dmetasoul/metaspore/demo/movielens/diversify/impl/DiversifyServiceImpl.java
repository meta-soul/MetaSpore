package com.dmetasoul.metaspore.demo.movielens.diversify.impl;

import com.dmetasoul.metaspore.demo.movielens.diversify.DiversifierService;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.DiverseProvider;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.Diversifier;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class DiversifyServiceImpl implements DiversifierService {
    public static final List<String> DIVERSIFIER_NAMES = List.of("SimpleDiversifier");

    private final DiverseProvider diverseProvider;

    public DiversifyServiceImpl(DiverseProvider diverseProvider) {
        this.diverseProvider = diverseProvider;
    }

    @Override
    public List<ItemModel> diverse(List<ItemModel> itemModels, Integer window, Integer tolerance) {
        List<ItemModel> finalItemModelDiverse = new ArrayList<>();

        for (Diversifier m : diverseProvider.getDiversifiers(DIVERSIFIER_NAMES)) {
            finalItemModelDiverse = m.diverse(itemModels, window, tolerance);
        }
        return finalItemModelDiverse;
    }
}
