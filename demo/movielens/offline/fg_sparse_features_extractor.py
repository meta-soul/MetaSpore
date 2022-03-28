from pyspark.sql import functions as F
from functools import reduce

def generate_spare_features(dataset, max_len=10, sep=u'\u0001'):
    def get_recent_items(kv_pairs, max_len=max_len):
        #def get_genre_union_set(genres):
        #    result_set = set()
        #    for g in genres:
        #        result_set = result_set.union(set(g.split('|')))
        #    return result_set

        # sort
        kv_pairs.sort(key=lambda x: x[1])
        # get recent list
        recent_items = []
        for i in range(0, len(kv_pairs)):
            current, hist_time, genre = kv_pairs[i]
            hist_list = [0] if i == 0 else reduce(lambda x, y:x+y, map(lambda x:[x[0]], kv_pairs[:i]))
            #genre_list = ['None'] if i == 0 else reduce(lambda x, y:x+y, map(lambda x:[x[2]], kv_pairs[i-1:i]))
            last_movie = str(hist_list[-1])
            last_genre = 'None' if i == 0 else kv_pairs[i-1][2]
            # get last max_len items
            hist_list = hist_list[-max_len:]
            hist_list = str.join(sep, map(str, hist_list))
            #genre_list = genre_list[-max_len:]
            #genre_list = str.join(sep, get_genre_union_set(genre_list))
            recent_items.append((hist_time, current, hist_list, last_movie, last_genre))

        return recent_items
    
    # label
    dataset = dataset.withColumn('label',  F.when(F.col('rating')> 0, 1).otherwise(0))
    
    # generate user recent behaviors features
    hist_item_list_df = dataset.filter(dataset['rating']>0).select('user_id','movie_id','timestamp', 'genre').distinct().rdd\
                               .map(lambda x: (x['user_id'], [(x['movie_id'], x['timestamp'], x['genre'])]))\
                               .reduceByKey(lambda x, y: x + y)\
                               .map(lambda x: (x[0], get_recent_items(x[1])))\
                               .flatMapValues(lambda x: x)\
                               .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4]))\
                               .toDF(['user_id', \
                                      'timestamp', \
                                      'movie_id', \
                                      'recent_movie_ids', \
                                      'last_movie', \
                                      'last_genre'])
    # merge features
    fg_result = dataset.alias('t1')\
                       .join(hist_item_list_df.alias('t2'), \
                             (F.col('t1.user_id')==F.col('t2.user_id')) & (F.col('t1.timestamp')==F.col('t2.timestamp')) & (F.col('t1.movie_id')==F.col('t2.movie_id')),
                             how='leftouter')\
                       .select('t1.label', \
                               't1.user_id', \
                               't1.gender', \
                               't1.age', \
                               't1.occupation', \
                               't1.zip', \
                               't1.movie_id', \
                               't2.recent_movie_ids', \
                               't1.genre', \
                               't1.rating',\
                               't2.last_movie', \
                               't2.last_genre', \
                               't1.timestamp')
    # replace sep in genre column
    fg_result = fg_result.withColumn('genre', F.regexp_replace('genre', '\|', sep))
    fg_result = fg_result.withColumn('last_genre', F.regexp_replace('last_genre', '\|', sep))
    print('Debug -- fg result sample:')
    fg_result.show(10)
    return fg_result
