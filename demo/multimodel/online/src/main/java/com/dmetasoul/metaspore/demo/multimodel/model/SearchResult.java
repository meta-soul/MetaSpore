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

package com.dmetasoul.metaspore.demo.multimodel.model;

public class SearchResult {
    private String searchQuery;

    private String searchResult;

    private SearchContext searchContext;

    public SearchResult() {
    }

    public SearchResult(String searchQuery, SearchContext searchContext) {
        this.searchQuery = searchQuery;
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

    @Override
    public String toString() {
        return "SearchResult{" +
                "searchQuery='" + searchQuery + '\'' +
                ", searchResult='" + searchResult + '\'' +
                ", searchContext=" + searchContext +
                '}';
    }
}
