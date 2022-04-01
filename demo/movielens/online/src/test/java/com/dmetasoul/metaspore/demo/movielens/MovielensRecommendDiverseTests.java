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

package com.dmetasoul.metaspore.demo.movielens;

import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl.MMRDiversifier;
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
            peek.setFinalRankingScore(r.nextDouble()*3+2);
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
        MMRDiversifier diversfier = new MMRDiversifier();
        for (int i = 0; i < 10; i++) {
            System.out.println("第" + (i + 1) + "轮测试");
            List<ItemModel> input = getInput();
            System.out.println(input);
            List temp = diversfier.diverse(input, 4, 4,0.7);
            System.out.println(temp);

        }
    }
}