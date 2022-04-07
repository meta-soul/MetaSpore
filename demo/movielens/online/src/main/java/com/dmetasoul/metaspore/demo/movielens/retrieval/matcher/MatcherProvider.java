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

package com.dmetasoul.metaspore.demo.movielens.retrieval.matcher;

import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class MatcherProvider {

    private final Map<String, Matcher> matcherMap;

    public MatcherProvider(List<Matcher> matchers) {
        this.matcherMap = new HashMap<>();
        matchers.forEach(x -> matcherMap.put(x.getClass().getSimpleName(), x));
    }
    public Matcher getMatcher(String name) {
        return matcherMap.get(name);
    }

    public Collection<Matcher> getMatchers(Collection<String> names) {
        if (names == null) {
            return Collections.emptyList();
        }

        ArrayList<Matcher> matchers = new ArrayList<>();
        for (String name : names) {
            matchers.add(getMatcher(name));
        }

        return matchers;
    }
}