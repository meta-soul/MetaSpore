package com.dmetasoul.metaspore.demo.multimodal.service;

import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.protobuf.ByteString;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface HfPreprocessorService {
    Map<String, ByteString> predictBlocking(String modelName, List<String> texts, Map<String, String> parameters) throws IOException;
    Map<String, ArrowTensor> pbToArrow(Map<String, ByteString> payload) throws IOException;
    List<List<Float>> getFloatVectorsFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap, String targetKey);
    List<List<Integer>> getIntVectorsFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap, String targetKey);
    Map<String, List<List<Float>>> getFloatPredictFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap);
    Map<String, List<List<Integer>>> getIntPredictFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap);
}
