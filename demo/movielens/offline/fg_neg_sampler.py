import numpy as np
from pyspark.sql.functions import lit, col, pow

# negative sampling on original dataset
def negative_sampling(spark, dataset, user_column='user_id', item_column='movie_id', time_column='timestamp', \
                      negative_item_column='trigger_item_id', negative_sample=3):
    
    def gen_sample_prob(dataset, group_by, alpha=0.75):
        item_weight = dataset.groupBy(col(group_by)).count()
        item_weight = item_weight.withColumn('norm_weight', pow(item_weight['count'], alpha))
        total_freq = item_weight.select('count').groupBy().sum().collect()[0][0]
        total_norm = item_weight.select('norm_weight').groupBy().sum().collect()[0][0]
        item_weight = item_weight.withColumn('sampling_prob', item_weight['norm_weight']/total_norm)    
        return item_weight, total_norm, total_freq
    
    def sample(user_id, user_item_list, item_list, dist_list, negative_sample):
        # sample negative list
        candidate_list = np.random.choice(list(item_list), size=len(user_item_list)*negative_sample, \
                                          replace=True, p=list(dist_list)).tolist()
        # remove the positive sample from the sampling result
        candidate_list = list(set(candidate_list)-set(user_item_list))
        
        # sample trigger list
        trigger_list = np.random.choice(list(user_item_list), size=len(candidate_list), \
                                        replace=True).tolist()
        
        return list(zip(trigger_list, candidate_list))
    
    # sampling distribution
    item_weight, _, _ = gen_sample_prob(dataset, item_column)
    # item_list = item_weight.select(item_column).rdd.flatMap(lambda x: x).collect()
    # dist_list = item_weight.select('sampling_prob').rdd.flatMap(lambda x: x).collect()
    ## ref: https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
    items_dist = item_weight.select('movie_id', 'sampling_prob')\
                            .rdd.map(lambda x: (x[0], x[1])).collect()
    zipped_dist = [list(t) for t in zip(*items_dist)]
    item_list, dist_list = zipped_dist[0], zipped_dist[1]
    # reference: https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
    dist_list = np.asarray(dist_list).astype('float64')
    dist_list = list(dist_list / np.sum(dist_list))
    print('Debug -- item_list.size:', len(item_list))
    print('Debug -- dist_list.size:', len(dist_list))
    ## broadcast ref:
    ## * https://stackoverflow.com/questions/68846636/relevance-of-spark-broadcast-variable-in-filter-a-data-frame-using-a-list-tuple
    ## * https://stackoverflow.com/questions/41045917/what-is-the-maximum-size-for-a-broadcast-object-in-spark
    ## * https://stackoverflow.com/questions/34770720/broadcast-a-dictionary-to-rdd-in-pyspark/34770804
    dist_list_br = spark.sparkContext.broadcast(dist_list)
    # generate sampling dataframe
    sampling_df=dataset.rdd\
                       .map(lambda x: (x[user_column], [x[item_column]]))\
                       .reduceByKey(lambda x, y: x + y)\
                       .map(lambda x: (x[0], sample(x[0], x[1], item_list, dist_list_br.value, negative_sample)))\
                       .flatMapValues(lambda x: x)\
                       .map(lambda x: (x[0], x[1][0], x[1][1]))\
                       .toDF([user_column, negative_item_column, item_column])
    
    return sampling_df

def negative_sampling_train_dataset(spark, train_fg_dataset, num_negs):
    # negative sampling
    neg_sample_df = negative_sampling(spark, dataset=train_fg_dataset, user_column='user_id', item_column='movie_id', \
                                      time_column='timestamp', negative_sample=num_negs)

    # merge into item and user profile information
    neg_sample_df = neg_sample_df.withColumn('label', lit(0))\
                                 .withColumn('rating', lit(0))
    neg_sample_df = neg_sample_df.alias('t1')\
                            .join(train_fg_dataset.alias('t2'), \
                                (col('t1.user_id')==col('t2.user_id')) & (col('t1.trigger_item_id')==col('t2.movie_id')),
                                how='leftouter')\
                            .select('t1.label', \
                                't1.user_id', \
                                't2.gender', \
                                't2.age', \
                                't2.occupation', \
                                't2.zip', \
                                't1.movie_id', \
                                't2.recent_movie_ids', \
                                't2.genre', \
                                't1.rating', \
                                't2.last_movie', \
                                't2.last_genre')

    # show negative sampling result
    print('Debug -- negative sampling result size:%d'%neg_sample_df.count())
    print('Debug -- negative samping result:')
    neg_sample_df.show(10)
    return neg_sample_df
