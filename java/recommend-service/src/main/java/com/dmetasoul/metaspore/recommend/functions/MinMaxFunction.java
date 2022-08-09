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
package com.dmetasoul.metaspore.recommend.functions;

import com.dmetasoul.metaspore.recommend.annotation.TransformFunction;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.serving.FeatureTable;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.FieldVector;

import java.util.List;
import java.util.Map;

@Slf4j
@TransformFunction("minmax")
public class MinMaxFunction extends Function {
    private final static String NAMEMIN = "min";
    private final static String NAMEMAX = "max";

    private int min = 0;
    private int max = 120;

    @Override
    public void init(Map<String, Object> params) {
    }

    @Override
    public List<Object> process(List<FieldData> fields, Map<String, Object> options) {
        return null;
    }
}
