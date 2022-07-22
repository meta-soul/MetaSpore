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
package com.dmetasoul.metaspore.recommend.datasource;

import com.dmetasoul.metaspore.recommend.annotation.DataSourceAnnotation;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;

import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataSourceAnnotation("request")
public class RequestSource extends DataSource {

    @Override
    public boolean initService() {
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        if (!source.getKind().equals("request")) {
            log.error("config request fail! is not kind:{} eq request!", source.getKind());
            return false;
        }
        return true;
    }

    @Override
    public void close() {}

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = new DataResult();
        result.setValues(context.getRequest());
        return result;
    }
}
