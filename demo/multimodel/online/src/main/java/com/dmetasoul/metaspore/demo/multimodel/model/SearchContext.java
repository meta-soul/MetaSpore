package com.dmetasoul.metaspore.demo.multimodel.model;

public class SearchContext {
    private String userId;

    public SearchContext() {

    }

    public SearchContext(String userId) {
        this.userId = userId;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    @Override
    public String toString() {
        return "SearchContext{" +
                "userId='" + userId + '\'' +
                '}';
    }
}
