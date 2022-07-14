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
    public boolean checkRequest(ServiceRequest request, DataContext context) {
        if (MapUtils.isEmpty(context.getRequest())) {
            log.error("request data must not be empty!");
            return false;
        }
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = new DataResult();
        result.setValues(context.getRequest());
        return result;
    }
}
