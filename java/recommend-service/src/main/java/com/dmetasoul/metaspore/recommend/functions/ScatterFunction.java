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

import com.dmetasoul.metaspore.recommend.data.FieldData;

import java.util.List;
import java.util.Map;

public abstract class ScatterFunction extends Function {

    @Override
    public List<Object> process(List<FieldData> fields, Map<String, Object> options) {
        throw new IllegalCallerException("ScatterFunction only has aggregate function!");
    }

    public abstract Map<String, List<Object>> scatter(List<FieldData> fields, List<String> names, Map<String, Object> options);

    public void init(Map<String, Object> params) {}
}
