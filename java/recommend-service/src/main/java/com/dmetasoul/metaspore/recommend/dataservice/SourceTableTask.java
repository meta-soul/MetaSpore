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
package com.dmetasoul.metaspore.recommend.dataservice;

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.datasource.DataSource;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;

import java.util.*;
import java.util.concurrent.*;

@SuppressWarnings("rawtypes")
@Slf4j
@DataServiceAnnotation("SourceTable")
public class SourceTableTask extends DataService {

    private DataSource dataSource;
    protected FeatureConfig.Source source;
    protected FeatureConfig.SourceTable sourceTable;

    @Override
    public boolean initService() {
        sourceTable = taskFlowConfig.getSourceTables().get(name);
        dataSource = taskServiceRegister.getDataSources().get(sourceTable.getSource());
        source = taskFlowConfig.getSources().get(sourceTable.getSource());
        return true;
    }

    @Override
    public boolean checkResult(DataResult result) {
        if (result == null) {
            log.warn("result is null!");
            return false;
        }
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(name);
        for (String col : sourceTable.getColumnNames()) {
            String type = sourceTable.getColumnMap().get(col);
            Class dataClass = DataTypes.getDataClass(type);
            if (MapUtils.isNotEmpty(result.getValues())) {
                Map<String, Object> data = result.getValues();
                Object value = data.get(col);
                if (value != null && !dataClass.isInstance(value)) {
                    log.warn("sourceTable {} get result col:{} type is wrong, value:{}", name, col, value);
                    // return false;
                }
            }
            if (CollectionUtils.isNotEmpty(result.getData())) {
                for (Map data : result.getData()) {
                    Object value = data.get(col);
                    if (value != null && !dataClass.isInstance(value)) {
                        log.warn("sourceTable {} get result col:{} type is wrong, value:{}", name, col, value);
                        // return false;
                    }
                }
            }
        }
        return true;
    }

    protected DataResult processRequest(ServiceRequest request, DataContext context) {
        return dataSource.process(request, context);
    }

    public <T> T getOptionOrDefault(String key, T value) {
        return Utils.getField(sourceTable.getOptions(), key, value);
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult dataResult;
        int retryNum = getOptionOrDefault("retryNum", 0);
        long timeOut = getOptionOrDefault("timeOut", 30000L);
        TimeUnit timeOutUnit = TimeUnit.MILLISECONDS;
        retryNum += 1;
        do {
            CompletableFuture<DataResult> future = CompletableFuture.supplyAsync(() -> {
                DataResult res = processRequest(request, context);
                if (checkResult(res)) {
                    return res;
                } else {
                    return null;
                }
            }, dataSource.getFeaturePool()).exceptionally(error -> {
                log.error("exec fail at {}, exception:{}!", name, error);
                return null;
            });
            try {
                dataResult = future.get(timeOut, timeOutUnit);
                if (checkResult(dataResult)) {
                    return dataResult;
                }
                retryNum -= 1;
            } catch (InterruptedException | ExecutionException e) {
                log.error("there was an error when executing the CompletableFuture",e);
            } catch (TimeoutException e) {
                log.error("when task timeout!",e);
            }
        } while (retryNum >= 0);
        return null;
    }
}
