package com.dmetasoul.metaspore.demo.movielens;

import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl.SimpleDiversifier;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import org.junit.jupiter.api.Test;

import java.util.*;

public class MovielensRecommendDiverseTests {
    String[] movie_genre = {
            "Drama",
            "Comedy",
            "Horror",
            "Documentary",
            "Thriller",
//                "War",
//                "Sci-Fi",
//                "Animation",
//                "Children's",
//                "Drama|Mystery",
            "Animation|Children's"};
    String[] movie_title = {
            "Man of Her Dreams",
            "Garden of Finz",
            "Blue Angel",
            "Sixth Man",
            "Make Them Die",
            "Poison Ivy",
            "Land Before",
            "Puppet Master",
            "Return of Jafar",
            "Hour of the Pig",
            "Slaughterhouse",
            "Bewegte Mann",
            "Celestial Clockwo",
            "Price of Glory",
            "Dangerous Beauty",
            "|Mountain Eagle",
            "Hour of the Pig",
            "Condition Red",
            "Where Eagles Dare",
            "|Aladdin and the K"
    };

    @Test
    public List<ItemModel> getInput() {
        List<ItemModel> input = new ArrayList<>();
        Random r = new Random();
        for (int i = 0; i < 15; i++) {
            Integer movie_id = r.nextInt(1000);
            ItemModel peek = new ItemModel();
            peek.setId(movie_id.toString());
            peek.setGenre(movie_genre[r.nextInt(movie_genre.length)]);
            peek.setTitle(movie_title[r.nextInt(movie_title.length)]);
            peek.setMovieAvgRating(r.nextDouble() * 5);
            input.add(peek);
        }
        Collections.sort(input, new Comparator<ItemModel>() {
            @Override
            public int compare(ItemModel o1, ItemModel o2) {
                Double diff = o1.getMovieAvgRating() - o2.getMovieAvgRating();
                if (diff > 0) {
                    return -1;
                } else {
                    return 1;
                }
            }
        });
        return input;
    }

    @Test
    public void testDiverse() {
        SimpleDiversifier diversfier = new SimpleDiversifier();
        for (int i = 0; i < 10; i++) {
            System.out.println("第" + (i + 1) + "轮测试");
            List<ItemModel> input = getInput();
            System.out.println(input);
            List temp = diversfier.diverse(input, 4, 4);
            System.out.println(temp);

        }
    }
}

