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

package com.dmetasoul.metaspore.demo.multimodal.algo.qp.processor;

import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class ProcessorProvider {
    private final Map<String, Processor> processorMap;

    public ProcessorProvider(List<Processor> processors) {
        this.processorMap = new HashMap<>();
        processors.forEach(x -> processorMap.put(x.getClass().getSimpleName(), x));
    }

    public Processor getProcessor(String name) {
        return processorMap.get(name);
    }

    public Collection<Processor> getProcessors(Collection<String> names) {
        if (names == null) {
            return Collections.emptyList();
        }

        ArrayList<Processor> processors = new ArrayList<>();
        for (String name : names) {
            processors.add(getProcessor(name));
        }

        return processors;
    }
}
