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

import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl.SimpleDiversifier;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import org.junit.jupiter.api.Test;

import javax.print.DocFlavor;
import java.util.*;

public class MovielensRecommendDiverseTests {
    String[] movie_genre = {
            "Drama",
            "Comedy",
            "Horror",
            "Horroru\001Comedy",
            "Documentaryu\001Thriller",
            "Documentaryu\001Comedy",
            "Dramau\001Comedy",
//            "Documentary",
//            "Thriller",
//                "War",
//                "Sci-Fi",
//                "Animation",
//                "Children's",
//                "Drama|Mystery",
            "Animationu\001Children's"};
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
            String genre=movie_genre[Math.min(movie_genre.length-1,r.nextInt(Math.max(1,movie_genre.length-1)))];
            peek.setGenre(genre);
            peek.setGenreList(genre);
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
        SimpleDiversifier diversfier = new SimpleDiversifier();
        for (int i = 0; i < 100; i++) {
            System.out.println("第" + (i + 1) + "轮测试");
            List<ItemModel> input = getInput();
            for (int j = 0; j < input.size(); j++) {
                System.out.print(input.get(j).getGenre()+" ");
            }
            RecommendContext recommendContext = new RecommendContext("0");
            recommendContext.setLamada(0.7);
            System.out.println();
            System.out.println("=============================================================================================" +
                    "===========================================================================================");
            List<ItemModel> temp = diversfier.diverse(recommendContext,input, 4, 4);
            for (int j = 0; j < temp.size(); j++) {
                System.out.print(temp.get(j).getGenre()+" ");
            }
            System.out.println();

        }
    }
}