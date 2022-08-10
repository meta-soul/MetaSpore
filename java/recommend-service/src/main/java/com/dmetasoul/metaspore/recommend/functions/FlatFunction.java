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
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import lombok.SneakyThrows;

import java.util.List;
import java.util.Map;

public interface FlatFunction extends Function {

    @Override
    default List<Object> process(List<FieldData> fields, Map<String, Object> options) {
        throw new IllegalCallerException("FlatFunction only has flat function!");
    }
    /**
     *
     * @param indexs 用于设置flat之后新结果数据与原来字段数据的映射关系 list 下标与结果数据下标一致， Integer值表示对应输入数据的下标
     * @param fields  用于flat函数的输入数据
     * @param options  配置的函数参数
     * @return  flat之后生成的结果数据
     * 函数执行完毕。如果indexs为empty， 则清空之前函数计算的缓存结果，接下来的函数计算只使用当前flat函数结果进行计算
     */
    List<Object> flat(List<Integer> indexs, List<FieldData> fields, Map<String, Object> options);

}
