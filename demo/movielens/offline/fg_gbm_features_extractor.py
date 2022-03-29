#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyspark.sql.functions import regexp_replace

def generate_gbm_features(spark, users, movies, ratings, sep=u'\u0001'):
    user_rating = generate_user_avg_rating(spark, ratings)
    user_rating_more_than_three_rate = generate_user_rating_more_than_three_percent(spark, ratings)
    user_info = user_rating_more_than_three_rate\
                            .join(user_rating, on=user_rating_more_than_three_rate.user_id==user_rating.user_id, how="leftouter")\
                            .drop(user_rating_more_than_three_rate.user_id)
    print('Debug -- gbm model user_info sample:')
    user_info.show(10)
    
    movie_rating = generate_movie_avg_rating_and_volume(spark, ratings)
    movie_rating_more_than_three_rate = generate_movie_rating_more_than_three_percent(spark, ratings)
    movie_info = movie_rating\
                    .join(movie_rating_more_than_three_rate, on=movie_rating.movie_id==movie_rating_more_than_three_rate.movie_id, how="leftouter")\
                    .drop(movie_rating.movie_id)
    print('Debug -- gbm model movie_info sample:')
    movie_info.show(10)

    genre_rating_avg_volume = generate_genre_avg_rating_and_volume(spark, ratings, movies)
    genre_dataset = generate_genre(spark, ratings, movies)
    genre_more_than_three_rating = generate_genre_more_than_three_rating(spark, genre_dataset)    
    genre_info = generate_genre_all_features(spark, genre_rating_avg_volume, genre_more_than_three_rating)
    genre_info = genre_info.withColumn('genre', regexp_replace('genre', '\|', sep))
    print('Debug -- gbm model genre_info sample:')
    genre_info.show(5)

    return  {'user': user_info, 'movie': movie_info, 'genre': genre_info}

def generate_user_avg_rating(spark, ratings):
    ratings.registerTempTable("ratings")
    query="""
    select 
        user_id, 
        avg(rating) as user_movie_avg_rating
    from ratings
    group by user_id
    """
    return spark.sql(query)

def generate_user_rating_more_than_three_percent(spark, ratings):
    ratings.registerTempTable("ratings")
    query="""
    select
        ta.user_id,
        coalesce(tb.user_rating_greater_than_three_count, 0)/(ta.user_rating_count + 0.1) as user_greater_than_three_rate
    from
    (
        select 
            user_id, 
            count(1) as user_rating_count
        from ratings
        group by user_id
    ) ta
    left outer join
    (
        select 
            user_id, 
            count(1) as user_rating_greater_than_three_count
        from ratings
        where rating>3
        group by user_id
    ) tb
    on ta.user_id=tb.user_id
    """
    return spark.sql(query)

def generate_movie_avg_rating_and_volume(spark, ratings):
    ratings.registerTempTable("ratings")
    query="""
    select
        movie_id,
        log(count(1)) as watch_volume,
        avg(rating) as movie_avg_rating
    from ratings
    group by movie_id
    """
    return  spark.sql(query)

def generate_movie_rating_more_than_three_percent(spark, ratings):
    ratings.registerTempTable("ratings")
    query="""
    select
        ta.movie_id,
        coalesce(tb.movie_rating_greater_than_three_count, 0)/(ta.movie_rating_count + 0.1) as movie_greater_than_three_rate
    from
    (
        select 
            movie_id, 
            count(1) movie_rating_count
        from ratings
        group by movie_id
    ) ta
    left outer join
    (
        select 
            movie_id,
            count(1) movie_rating_greater_than_three_count
        from ratings
        where rating>3
        group by movie_id
    ) tb
    on ta.movie_id=tb.movie_id
    """

    return spark.sql(query)

def generate_genre(spark, ratings, movies):
    ratings.registerTempTable("ratings")
    movies.registerTempTable("movies")
    query="""
    select 
        movies.movie_id,
        ratings.rating,
        movies.genre
    from 
        movies, ratings
    where 
        ratings.movie_id=movies.movie_id
    """
    return spark.sql(query)

def generate_genre_avg_rating_and_volume(spark, ratings, movies):
    ratings.registerTempTable("ratings")
    movies.registerTempTable("movies")
    query="""
    select 
        genre,
        log(count(*)) as genre_watch_volume,
        avg(rating) as genre_movie_avg_rating 
    from 
    (
        select 
            movies.movie_id,
            ratings.rating,
            movies.genre
        from 
            movies, ratings
        where 
            ratings.movie_id=movies.movie_id
    ) ta
    group by genre
    """
    return spark.sql(query)

def generate_genre_more_than_three_rating(spark, genre_dataset):
    genre_dataset.registerTempTable("genre")
    query="""
    select
        ta.genre,
        coalesce(tb.genre_rating_more_than_three_count, 0)/(ta.genre_rating_count + 0.1) as genre_greater_than_three_rate
    from
    (
        select 
            genre.genre, 
            count(genre.genre) genre_rating_count
        from genre
        group by genre.genre
    ) ta
    left join
    (
        select 
            genre.genre, 
            count(genre.genre) genre_rating_more_than_three_count
        from genre
        where genre.rating>3
        group by genre.genre
    ) tb
    on ta.genre=tb.genre
    """
    return spark.sql(query)

def generate_genre_all_features(spark, genre_rating_avg_volume, genre_more_than_three_rating):
    genre_rating_avg_volume.registerTempTable("genre_avg")
    genre_more_than_three_rating.registerTempTable("genre_morethan")
    query="""
    select
        genre_avg.genre,
        genre_avg.genre_watch_volume,
        genre_avg.genre_movie_avg_rating,
        genre_morethan.genre_greater_than_three_rate
    from 
        genre_avg, genre_morethan
    where
        genre_avg.genre=genre_morethan.genre
    """
    return spark.sql(query)
