//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
package com.dmetasoul.metaspore.recommend.data;

import com.dmetasoul.metaspore.recommend.enums.ResultTypeEnum;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.milvus.response.SearchResultsWrapper;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.Collection;
import java.util.List;
import java.util.Map;
/**
 * 用于保存服务结果
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
public class DataResult {
    String name;
    String reqSign;
    @Getter
    ResultTypeEnum resultType;
    Map<String, Object> values;
    List<Map> data;
    Map<Integer, List<SearchResultsWrapper.IDScore>> milvusData;

    FeatureArray featureArray;

    FeatureTable featureTable;

    PredictResult predictResult;

    @Data
    public static class PredictResult {
        ResultType type = ResultType.Other;
        private List<List<Float>> embedding;
        private List<Float> score;

        public List<List<Float>> getEmbedding() {
            if (type.equals(ResultType.Embedding)) {
                return embedding;
            }
            return null;
        }
        public List<Float> getScore() {
            if (type.equals(ResultType.Score)) {
                return  score;
            }
            return null;
        }

        public void setEmbedding(List<List<Float>> embedding) {
            if (embedding == null) {
                return;
            }
            this.embedding = embedding;
            type = ResultType.Embedding;
        }
        public void setScore(List<Float> score) {
            if (score == null) {
                return;
            }
            this.score = score;
            type = ResultType.Score;
        }

        enum ResultType {
            Embedding(0,"embedding"),
            Score(1, "score"),
            Other(10, "other");
            private Integer id;
            private String name;

            ResultType(int id, String name){
                this.id = id;
                this.name = name;
            }
        }
    }

    @Data
    public static class FeatureArray {
        Map<String, List<Object>> arrays;

        int maxIndex;
        public FeatureArray(Map<String, List<Object>> arrays) {
            if (arrays == null) {
                arrays = Maps.newHashMap();
            }
            this.maxIndex = 0;
            this.arrays = arrays;
            arrays.forEach((k,v) -> {
                if (v.size() > maxIndex) maxIndex = v.size();
            });
        }

        public boolean isInvalidIndex(int index) {
            return index < 0 || index >= maxIndex;
        }

        public Object get(String fieldName, int index) {
            if (isInvalidIndex(index)) return null;
            if (arrays.containsKey(fieldName)) {
                List<Object> data = arrays.get(fieldName);
                if (index >= data.size()) return null;
                return data.get(index);
            }
            return null;
        }

        public boolean inArray(String fieldName) {
            return arrays.containsKey(fieldName);
        }
        public List<Object> getArray(String fieldName) {
            return arrays.get(fieldName);
        }

        public Object get(String fieldName) {
            return get(fieldName, 0);
        }
    }
    public DataResult() {
        resultType = ResultTypeEnum.EMPTY;
        reqSign = "";
    }

    public boolean isVaild() {
        return resultType != ResultTypeEnum.EMPTY && resultType != ResultTypeEnum.EXCEPTION;
    }

    public static DataResult merge(List<DataResult> dataResults, String name) {
        DataResult result = null;
        if (CollectionUtils.isEmpty(dataResults)) return result;
        String dataName = name;
        if (StringUtils.isEmpty(dataName)) dataName = dataResults.get(0).getName();
        result = new DataResult();
        List<Map> data = Lists.newArrayList();
        for (DataResult item : dataResults) {
            Map<String, Object> map = item.getValues();
            if (map != null) {
                data.add(map);
                continue;
            }
            List<Map> list = item.getData();
            if (list != null) {
                data.addAll(list);
            }
        }
        result.setData(data);
        return result;
    }

    public void setValues(Map<String, Object> values) {
        resultType = ResultTypeEnum.VALUES;
        this.values = values;
    }

    public Map<String, Object> getValues() {
        if(resultType == ResultTypeEnum.VALUES) {
            return this.values;
        }
        return null;
    }

    public void setData(List<Map> data) {
        resultType = ResultTypeEnum.DATA;
        this.data = data;
    }

    public List<Map> getData() {
        if(resultType == ResultTypeEnum.DATA) {
            return this.data;
        }
        return null;
    }

    public void setMilvusData(Map<Integer, List<SearchResultsWrapper.IDScore>> milvusData) {
        resultType = ResultTypeEnum.MILVUS;
        this.milvusData = milvusData;
    }

    public void setFeatureArray(FeatureArray featureArray) {
        resultType = ResultTypeEnum.FEATUREARRAYS;
        this.featureArray = featureArray;
    }

    public void setFeatureArray(Map<String, List<Object>> arrays) {
        resultType = ResultTypeEnum.FEATUREARRAYS;
        this.featureArray = new FeatureArray(arrays);
    }

    public FeatureArray getFeatureArray() {
        if(resultType == ResultTypeEnum.FEATUREARRAYS) {
            return this.featureArray;
        }
        return null;
    }

    public void setFeatureTable(FeatureTable featureTable) {
        resultType = ResultTypeEnum.FEATURETABLE;
        this.featureTable = featureTable;
    }

    public FeatureTable getFeatureTable() {
        if(resultType == ResultTypeEnum.FEATURETABLE) {
            return this.featureTable;
        }
        return null;
    }

    public void setPredictResult(PredictResult predictResult) {
        resultType = ResultTypeEnum.PREDICTRESULT;
        this.predictResult = predictResult;
    }

    public PredictResult getPredictResult() {
        if(resultType == ResultTypeEnum.PREDICTRESULT) {
            return this.predictResult;
        }
        return null;
    }

}
