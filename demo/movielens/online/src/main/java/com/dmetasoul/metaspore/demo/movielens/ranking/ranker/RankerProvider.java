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

package com.dmetasoul.metaspore.demo.movielens.ranking.ranker;

import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class RankerProvider {

    private final Map<String, Ranker> rankerMap;

    public RankerProvider(List<Ranker> rankers) {
        this.rankerMap = new HashMap<>();
        rankers.forEach(x -> rankerMap.put(x.getClass().getSimpleName(), x));
    }

    public Ranker getRanker(String name) {
        return rankerMap.get(name);
    }

    public Collection<Ranker> getRankers(Collection<String> names) {
        if (names == null) {
            return Collections.emptyList();
        }

        ArrayList<Ranker> rankers = new ArrayList<>();
        for (String name : names) {
            rankers.add(getRanker(name));
        }

        return rankers;
    }
}