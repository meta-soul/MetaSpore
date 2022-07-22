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
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataSourceAnnotation("milvus")
public class MilvusSource extends DataSource {
    private MilvusServiceClient milvusTemplate;

    @Override
    public boolean initService() {
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        if (!source.getKind().equals("milvus")) {
            log.error("config milvus fail! is not kind:{} eq milvus!", source.getKind());
            return false;
        }
        String host = (String) source.getOptions().getOrDefault("host", "localhost");
        int port = (Integer) source.getOptions().getOrDefault("port", 9000);
        ConnectParam connectParam = ConnectParam.newBuilder()
                .withHost(host)
                .withPort(port)
                .build();
        milvusTemplate = new MilvusServiceClient(connectParam);
        return true;
    }

    @Override
    public void close() {
        milvusTemplate.close();
    }
}
