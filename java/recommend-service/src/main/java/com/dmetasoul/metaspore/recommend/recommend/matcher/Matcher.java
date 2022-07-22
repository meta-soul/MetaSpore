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

package com.dmetasoul.metaspore.recommend.recommend.matcher;

import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.dataservice.DataService;
import com.dmetasoul.metaspore.recommend.recommend.Service;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;

import java.util.List;

@Slf4j
@RecommendAnnotation("Matcher")
public class Matcher extends Service {
    public static final int DEFAULT_MAX_RESERVATION = 50;
    protected int maxReservation;

    protected DataService dataService;

    @Override
    public boolean initService() {
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        List<String> depend = serviceConfig.getDepend();
        if (CollectionUtils.isEmpty(depend) || depend.size() > 1) {
            log.error("matcher service depend config error");
            return false;
        }
        dataService = serviceRegister.getDataServices().get(depend.get(0));
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        ServiceRequest req = new ServiceRequest(request, context);
        req.setLimit(maxReservation);
        return dataService.execute(req, context);
    }
}