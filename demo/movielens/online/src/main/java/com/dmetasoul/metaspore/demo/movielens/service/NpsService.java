package com.dmetasoul.metaspore.demo.movielens.service;

import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.FeatureTable;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface NpsService {
    Map<String, ArrowTensor> predictBlocking(String modelName, Iterable<FeatureTable> featureTables, Map<String, String> parameters) throws IOException;
    List<List<Float>> getVectorsFromNpsResult(Map<String, ArrowTensor> nspResultMap, String targetKey);
    List<Float> getScoresFromNpsResult(Map<String, ArrowTensor> nspResultMap, String targetKey, int targetIndex);
}
