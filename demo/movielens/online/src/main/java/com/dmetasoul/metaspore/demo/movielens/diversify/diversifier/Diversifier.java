package com.dmetasoul.metaspore.demo.movielens.diversify.diversifier;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;

import java.util.List;

public interface Diversifier {
    List<ItemModel> diverse(List<ItemModel> itemModels, Integer window, Integer tolerance);
}
