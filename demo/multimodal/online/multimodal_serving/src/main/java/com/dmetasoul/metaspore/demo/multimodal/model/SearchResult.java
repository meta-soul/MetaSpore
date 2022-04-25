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

package com.dmetasoul.metaspore.demo.multimodal.model;

import com.dmetasoul.metaspore.demo.multimodal.model.QueryModel;
import com.dmetasoul.metaspore.demo.multimodal.model.ItemModel;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.google.protobuf.ByteString;

import java.util.List;
import java.util.Map;

public class SearchResult {
    @JsonIgnore
    private String searchQuery;

    @JsonIgnore
    private String searchResult;

    @JsonIgnore
    private SearchContext searchContext;

    private QueryModel queryModel;

    private List<List<ItemModel>> searchItemModels;

    public SearchResult() {
    }

    public SearchResult(String searchQuery, SearchContext searchContext) {
        this.searchQuery = searchQuery;
        this.searchContext = searchContext;
        this.queryModel = new QueryModel();
        this.queryModel.setQuery(searchQuery);
    }

    public SearchResult(QueryModel queryModel, SearchContext searchContext) {
        this.queryModel = queryModel;
        this.searchContext = searchContext;
    }

    public String getSearchQuery() {
        return searchQuery;
    }

    public void setSearchQuery(String searchQuery) {
        this.searchQuery = searchQuery;
    }

    public String getSearchResult() {
        return searchResult;
    }

    public void setSearchResult(String searchResult) {
        this.searchResult = searchResult;
    }

    public SearchContext getSearchContext() {
        return searchContext;
    }

    public void setSearchContext(SearchContext searchContext) {
        this.searchContext = searchContext;
    }

    public QueryModel getQueryModel() { return queryModel; }

    public void setQueryModel(QueryModel queryModel) { this.queryModel = queryModel; }

    public List<List<ItemModel>> getSearchItemModels() { return searchItemModels; }

    public void setSearchItemModels(List<List<ItemModel>> itemModels) { this.searchItemModels = itemModels; }

    @Override
    public String toString() {
        return "SearchResult{" +
                "searchQueryModel='" + queryModel + '\'' +
                ", searchResult='" + searchResult + '\'' +
                ", searchContext=" + searchContext +
                '}';
    }
}
