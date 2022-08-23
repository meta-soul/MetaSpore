package com.dmetasoul.metaspore.recommend.recommend.interfaces;

import java.util.Map;

public interface MergeOperator {
    Object merge(Object field, Object data, Map<String, Object> option);
}
