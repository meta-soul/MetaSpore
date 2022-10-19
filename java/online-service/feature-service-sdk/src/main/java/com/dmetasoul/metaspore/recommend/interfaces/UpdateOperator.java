package com.dmetasoul.metaspore.recommend.interfaces;

import java.util.List;
import java.util.Map;

public interface UpdateOperator {
    Map<String, Object> update(List<Object> input, List<String> output, Map<String, Object> option);
}
