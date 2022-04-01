package com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl;


import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;

import java.util.*;
public class MMRDiversifier {
    public static List<ItemModel> diverse(List<ItemModel> itemModels,
                                          Integer window,
                                          Integer tolerance,
                                          Double lamada) {
        int genreCount = groupByType(itemModels).size();
        if (window == null || window > genreCount) {
            window = genreCount;
        }
        //label the visited
        HashMap<ItemModel,Integer> itemVisited=new HashMap<>();
        for (int i = 0; i < itemModels.size(); i++) {
            itemVisited.put(itemModels.get(i),0);
        }
        //compute the genre in the window
        HashMap<String,Integer> itemInWindow=new HashMap();
        //start diverse
        for (int i = 0; i < itemModels.size(); i++) {
            int num=0;
            //copmpute the count of genre in window
            if(!itemInWindow.isEmpty()){
                for(String genre: itemInWindow.keySet()){
                    num+=itemInWindow.get(genre);
                }
            }

            if(itemInWindow.containsKey(itemModels.get(i).getGenre())){
                int minindex=0;
                double minMMR=Double.MAX_VALUE;

                for (int j =i; j <Math.min(i+tolerance,itemModels.size()); j++) {
                    if(itemVisited.get(itemModels.get(j))!=0){
                        continue;
                    }
                    double rankingScore=itemModels.get(j).getFinalRankingScore()*lamada;
                    double simRate=getsim(itemModels.get(j),itemInWindow)*(1-lamada);
                    if((rankingScore+simRate)<minMMR){
                        minindex=j;
                        minMMR=rankingScore+simRate;
                    }
                }
                String minGenre= itemModels.get(minindex).getGenre();
                int defaults=itemInWindow.containsKey(minGenre)?itemInWindow.get(minGenre)+1:1;
                itemInWindow.put(minGenre,defaults);
                ItemModel needDiverse=itemModels.get(minindex);
                itemVisited.put(itemModels.get(minindex),1);
                for (int j = minindex; j >i; j--) {
                    itemModels.set(j,itemModels.get(j-1));
                }
                itemModels.set(i,needDiverse);
            }else{
                itemInWindow.put(itemModels.get(i).getGenre(),1);
                itemVisited.put(itemModels.get(i),1);
            }if(num==window){
                String deleteGenre=itemModels.get(i-window+1).getGenre();
                itemInWindow.put(deleteGenre,itemInWindow.get(deleteGenre)-1);
                if(itemInWindow.get(deleteGenre)==0){
                    itemInWindow.remove(deleteGenre);
                }
            }
        }

        return itemModels;
    }
    public static Double getsim(ItemModel item,HashMap<String,Integer> itemInWindow){
        HashSet<String> genreSet=new HashSet<>();
        for(String genre:itemInWindow.keySet()){
            String [] genreTemp=genre.split("u\0001");
            Set<String> set=new HashSet<String>(Arrays.asList(genreTemp));
            genreSet.addAll(set);
        }

        if(itemInWindow.containsKey(item.getGenre())){
            double genreLength=item.getGenre().split("u\0001").length;
            return genreLength/genreSet.size();
        }else{
            return 0.0;
        }
    }

    public static Map<String, List<ItemModel>> groupByType(List<ItemModel> numbers) {
        Map<String, List<ItemModel>> map = new HashMap<>();
        for (ItemModel item : numbers) {
            if (map.containsKey(item.getGenre())) {
                map.get(item.getGenre()).add(item);
            } else {
                List<ItemModel> ls = new ArrayList<>();
                ls.add(item);
                map.put(item.getGenre(), ls);
            }
        }
        return map;
    }
}
