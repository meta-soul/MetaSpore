package com.dmetasoul.metaspore.demo.movielens.model;

import com.dmetasoul.metaspore.demo.movielens.domain.Item;
import com.dmetasoul.metaspore.demo.movielens.domain.ItemFeature;

import java.util.HashMap;
import java.util.Optional;

public class ItemModel {
    private String id;
    private String title;
    private String genre;
    private String imdbUrl;
    private Double genreGreaterThanThreeRate;
    private Double genreMovieAvgRating;
    private Double genreWatchVolume;
    private Double movieAvgRating;
    private Double movieGreaterThanThreeRate;
    private Double watchVolume;
    private final HashMap<String, Double> originalRetrievalScoreMap;
    private Double finalRetrievalScore;
    private final HashMap<String, Double> originalRankingScoreMap;
    private Double finalRankingScore;

    public ItemModel() {
        originalRetrievalScoreMap = new HashMap<>();
        originalRankingScoreMap = new HashMap<>();
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getGenre() {
        return genre;
    }

    public void setGenre(String genre) {
        this.genre = genre;
    }

    public String getImdbUrl() {
        return imdbUrl;
    }

    public void setImdbUrl(String imdbUrl) {
        this.imdbUrl = imdbUrl;
    }

    public Double getGenreGreaterThanThreeRate() {
        return genreGreaterThanThreeRate;
    }

    public void setGenreGreaterThanThreeRate(Double genreGreaterThanThreeRate) {
        this.genreGreaterThanThreeRate = genreGreaterThanThreeRate;
    }

    public Double getGenreMovieAvgRating() {
        return genreMovieAvgRating;
    }

    public void setGenreMovieAvgRating(Double genreMovieAvgRating) {
        this.genreMovieAvgRating = genreMovieAvgRating;
    }

    public Double getGenreWatchVolume() {
        return genreWatchVolume;
    }

    public void setGenreWatchVolume(Double genreWatchVolume) {
        this.genreWatchVolume = genreWatchVolume;
    }

    public Double getMovieAvgRating() {
        return movieAvgRating;
    }

    public void setMovieAvgRating(Double movieAvgRating) {
        this.movieAvgRating = movieAvgRating;
    }

    public Double getMovieGreaterThanThreeRate() {
        return movieGreaterThanThreeRate;
    }

    public void setMovieGreaterThanThreeRate(Double movieGreaterThanThreeRate) {
        this.movieGreaterThanThreeRate = movieGreaterThanThreeRate;
    }

    public Double getWatchVolume() {
        return watchVolume;
    }

    public void setWatchVolume(Double watchVolume) {
        this.watchVolume = watchVolume;
    }

    public Double getFinalRetrievalScore() {
        return finalRetrievalScore;
    }

    public void setFinalRetrievalScore(Double finalRetrievalScore) {
        this.finalRetrievalScore = finalRetrievalScore;
    }

    public Double getFinalRankingScore() {
        return finalRankingScore;
    }

    public void setFinalRankingScore(Double finalRankingScore) {
        this.finalRankingScore = finalRankingScore;
    }

    public HashMap<String, Double> getOriginalRetrievalScoreMap() {
        return originalRetrievalScoreMap;
    }

    public HashMap<String, Double> getOriginalRankingScoreMap() {
        return originalRankingScoreMap;
    }

    public void fillSummary(Item item) {
        this.title = item.getTitle();
        this.genre = item.getGenre();
        this.imdbUrl = item.getImdbUrl();
    }

    public void fillFeatures(ItemFeature itemFeature) {
        this.genre = itemFeature.getGenre();
        this.genreGreaterThanThreeRate = Optional.ofNullable(itemFeature.getGenreGreaterThanThreeRate()).orElse(0.0);
        this.genreMovieAvgRating = Optional.ofNullable(itemFeature.getGenreMovieAvgRating()).orElse(0.0);
        this.genreWatchVolume = Optional.ofNullable(itemFeature.getGenreWatchVolume()).orElse(0.0);
        this.movieAvgRating = Optional.ofNullable(itemFeature.getMovieAvgRating()).orElse(0.0);
        this.movieGreaterThanThreeRate = Optional.ofNullable(itemFeature.getMovieGreaterThanThreeRate()).orElse(0.0);
        this.watchVolume = Optional.ofNullable(itemFeature.getWatchVolume()).orElse(0.0);
    }

}
