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

import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.milvus.response.SearchResultsWrapper;
import lombok.Data;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;
/**
 * 用于保存服务结果
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
public class DataResult {
    protected String name;
    protected String reqSign;
    protected FeatureTable featureTable;

    public List<Object> get(String field) {
        if (featureTable == null || featureTable.getVector(field) == null) return Lists.newArrayList();
        FieldVector vector = featureTable.getVector(field);
        List<Object> values = Lists.newArrayList();
        for (int i = 0; i < vector.getValueCount(); ++i) {
            values.add(vector.getObject(i));
        }
        return values;
    }

    public void setFeatureTable(FeatureTable featureTable) {
        this.featureTable = featureTable;
    }

    public boolean isNull() {
        return featureTable == null;
    }
}
