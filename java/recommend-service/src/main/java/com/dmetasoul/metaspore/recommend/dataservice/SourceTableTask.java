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
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
/**
 * SourceTable的DataService的实现类
 * SourceTable用于调用source数据源信息请求相关数据，sourcetable配置定义了数据的schema和部分条件
 * 注解DataServiceAnnotation 必须设置， value应设置为SourceTable。
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
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
        for (String col: sourceTable.getColumnNames()) {
            DataTypeEnum dataType = DataTypes.getDataType(sourceTable.getColumnMap().get(col));
            resFields.add(Field.nullable(col, dataType.getType()));
            dataTypes.add(dataType);
        }
        return true;
    }

    protected List<Map<String, Object>> processRequest(ServiceRequest request, DataContext context) {
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
                List<Map<String, Object>> res = processRequest(request, context);
                DataResult result = setDataResult(res);
                if (checkResult(result)) {
                    return result;
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
