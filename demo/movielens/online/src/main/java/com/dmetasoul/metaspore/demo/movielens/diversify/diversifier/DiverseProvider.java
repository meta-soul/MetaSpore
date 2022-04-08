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

package com.dmetasoul.metaspore.demo.movielens.diversify.diversifier;

import org.springframework.stereotype.Component;

import java.util.*;

@Component
public class DiverseProvider {


    private final Map<String, Diversifier> diversifierMap;

    public DiverseProvider(List<Diversifier> diversifiers) {
        this.diversifierMap = new HashMap<>();
        diversifiers.forEach(x -> diversifierMap.put(x.getClass().getSimpleName().toLowerCase(), x));
    }

    public Diversifier getDiversifiers(String name) {
        if (name == null) {
            return null;
        }
        return diversifierMap.get(name.toLowerCase());
    }
}