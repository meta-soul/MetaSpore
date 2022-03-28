package com.dmetasoul.metaspore.demo.movielens.diversify;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;

import java.util.List;

public interface DiversifierService {
    List<ItemModel> diverse(List<ItemModel> itemModels,
                            Integer window,
                            Integer tolerance);
}
