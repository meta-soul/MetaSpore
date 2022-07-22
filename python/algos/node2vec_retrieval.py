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

import pyspark.ml.base
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.ml.feature import Word2Vec, BucketedRandomProjectionLSH, VectorAssembler, MinMaxScaler

class Node2VecModel(pyspark.ml.base.Model):
    def __init__(self,
                 df=None,
                 key_column_name='key',
                 value_column_name='value',
                 vertex_score_delimiter=':',
                 vertex_score_pair_delimiter=';',
                 trigger_vertex_column_name=None,
                 debug=False):
        super().__init__()
        self.df = df
        self.key_column_name = key_column_name
        self.value_column_name = value_column_name
        self.vertex_score_delimiter = vertex_score_delimiter
        self.vertex_score_pair_delimiter = vertex_score_pair_delimiter
        self.trigger_vertex_column_name = trigger_vertex_column_name
        self.debug = debug
        
    def _transform(self, dataset):
        if self.trigger_vertex_column_name is None:
            raise ValueError("trigger_vertex_column_name is required")
        on = dataset[self.trigger_vertex_column_name] == self.df[self.key_column_name]     
        return dataset.join(self.df, on=on, how='left_outer')
    
    def _format_delimiter(self, string):
        return ''.join('\\u%04X' % ord(c) for c in string)

    def _get_value_expr(self):
        string = "array_join(transform(%s, " % self.value_column_name
        string += "t -> concat(t._1, '%s', t._2)" % self._format_delimiter(self.vertex_score_delimiter)
        string += "), '%s') " % self._format_delimiter(self.vertex_score_pair_delimiter)
        string += "AS %s" % self.value_column_name
        return string

    def stringify(self):
        key = self.key_column_name
        value = self._get_value_expr()
        self.df = self.df.selectExpr(key, value)
        return self
    
    def publish(self):
        pass


class Node2VecEstimator(pyspark.ml.base.Estimator):
    def __init__(self,
                 source_vertex_column_name=None,
                 destination_vertex_column_name=None,
                 weight_column_name=None,
                 trigger_vertex_column_name=None,
                 behavior_column_name=None,
                 behavior_filter_value=None,
                 max_recommendation_count=20,
                 random_walk_p=2.0,
                 random_walk_q=0.5,
                 random_walk_Z=1.0,
                 random_walk_steps=10,
                 walk_times=8,
                 key_column_name='key',
                 value_column_name='value',
                 vertex_score_delimiter=':',
                 vertex_score_pair_delimiter=';',
                 w2v_vector_size=5,
                 w2v_window_size=30,
                 w2v_min_count=0,
                 w2v_max_iter=10,
                 w2v_num_partitions=1,
                 euclid_bucket_length=100,
                 euclid_distance_threshold=10,
                 debug=False):
        super().__init__()
        self.source_vertex_column_name = source_vertex_column_name
        self.destination_vertex_column_name = destination_vertex_column_name
        self.weight_column_name = weight_column_name
        self.trigger_vertex_column_name = trigger_vertex_column_name
        self.behavior_column_name = behavior_column_name
        self.behavior_filter_value = behavior_filter_value
        self.max_recommendation_count = max_recommendation_count
        self.random_walk_p = random_walk_p
        self.random_walk_q = random_walk_q
        self.random_walk_Z = random_walk_Z
        self.random_walk_steps = random_walk_steps
        self.walk_times = walk_times
        self.key_column_name = key_column_name
        self.value_column_name = value_column_name
        self.vertex_score_delimiter = vertex_score_delimiter
        self.vertex_score_pair_delimiter = vertex_score_pair_delimiter
        self.debug = debug
        self.vertices_lookup = None
        self.edges_lookup = None
        self.w2v_vector_size = w2v_vector_size
        self.w2v_window_size = w2v_window_size
        self.w2v_min_count = w2v_min_count
        self.w2v_max_iter = w2v_max_iter
        self.w2v_num_partitions = w2v_num_partitions
        self.euclid_bucket_length = euclid_bucket_length
        self.euclid_distance_threshold = euclid_distance_threshold
        

    @staticmethod
    def setup_alias(weights):
        from collections import deque
        
        N = len(weights)
        p = [-1.0] * N
        a = [-1] * N
        small = deque()
        large = deque()

        summation = sum(weights)
        for idx, weight in enumerate(weights):
            p[idx] = N * weight / summation
            small.append(idx) if p[idx] < 1.0 else large.append(idx)  

        while len(small) > 0 and len(large) > 0:
            s = small.pop()
            l = large.pop()
            a[s] = l
            p[l] = p[l] + p[s] - 1.0
            small.append(l) if p[l] < 1.0 else large.append(l)

        while len(large) > 0:
            p[large.pop()] = 1.0

        while len(small) > 0:
            p[small.pop()] = 1.0

        return p, a

    @staticmethod
    def draw_alias(p, a):
        from random import Random
        from time import time
        from math import floor

        rdg = Random(time())
        idx = floor(rdg.random() * len(p))
        return idx if rdg.random() < p[idx] else a[idx]
    
    @staticmethod
    def verify(weights, p, a, sample_numb = 10000):
        N = len(weights)
        S = sum(weights)
        origin_probs = []
        for w in weights:
            origin_probs.append(w / S)
        print('Debug - original probs: ', origin_probs)

        count = [0] * N
        for i in range(sample_numb):
            idx = Node2VecEstimator.draw_alias(p, a)
            count[idx] = count[idx] + 1
        print('Debug - sampled probs: ', [c / sample_numb for c in count])
            
    def _filter_dataset(self, dataset):
        if self.behavior_column_name is None and self.behavior_filter_value is None:
            return dataset
        if self.behavior_column_name is not None and self.behavior_filter_value is not None:
            return dataset.where(dataset[self.behavior_column_name] == self.behavior_filter_value)
        
        raise RuntimeError("behavior_column_name and behavior_filter_value must be neither set or both set")
    
    def _preprocess_dataset(self, dataset):
        if self.source_vertex_column_name is None:
            raise ValueError("source_vertex_column_name is required")
        if self.destination_vertex_column_name is None:
            raise ValueError("destination_vertex_column_name is required")
        if self.trigger_vertex_column_name is None:
            raise ValueError("trigger_vertex_column_name is required")
        
        if self.weight_column_name is None:
            return dataset.select(F.col(self.source_vertex_column_name).alias("src"), 
                                  F.col(self.destination_vertex_column_name).alias("dst"),
                                  F.lit(1.0).alias("weight"))
        else:
            return dataset.select(F.col(self.source_vertex_column_name).alias("src"), 
                                  F.col(self.destination_vertex_column_name).alias("dst"),
                                  F.col(self.weight_column_name).alias("weight"))
      
    def _create_model(self, df):
        model = Node2VecModel(df=df,
                              key_column_name=self.key_column_name,
                              value_column_name=self.value_column_name,
                              vertex_score_delimiter=self.vertex_score_delimiter,
                              vertex_score_pair_delimiter=self.vertex_score_pair_delimiter,
                              trigger_vertex_column_name=self.trigger_vertex_column_name)
        return model
    
    def _init_vertices_lookup_df(self, edges):
        def _setup_vertices(row):
            src, attributes = row['src'], row['attributes']

            neighbors = []
            weights = []
            for attribute in attributes:
                neighbors.append(attribute['dst'])
                weights.append(attribute['weight'])

            p, a = Node2VecEstimator.setup_alias(weights)
            new_attributes = Row(neighbors=neighbors, p=p, a=a)

            return src, new_attributes
        
        if self.debug:
            print('Debug - edges:')
            edges.show(10, False)
        
        df = edges.groupBy(F.col('src')).agg(F.collect_list(F.struct(F.col('dst'),\
                                                                     F.col('weight'))).alias('attributes'))
        if self.debug:
            print('Debug - attributes of vertices:')
            df.show(10, False)
            df.printSchema()
        
        self.vertices_lookup = df.rdd.map(lambda row: _setup_vertices(row)).toDF(['src', 'attributes'])
        if self.debug:
            print('Debug - vertices_lookup:')
            self.vertices_lookup.show(10, False)
            self.vertices_lookup.printSchema()
    
    def _init_edges_lookup_df(self, edges):
        random_walk_p, random_walk_q, random_walk_Z = self.random_walk_p, self.random_walk_q, self.random_walk_Z
        def _setup_edges(row):
            src, dst, attributes = row['src'], row['dst'], row['attributes']
            dst_neighbors, src_neighbors = attributes['dst_neighbors'], attributes['src_neighbors']

            new_dst_neighbors = []
            pq_weights = []
            for dst_neighbor in dst_neighbors:
                neighbor_dst, neighbor_weight = dst_neighbor['dst'], dst_neighbor['weight']
                alpha = 1 / random_walk_q
                if neighbor_dst in src_neighbors:
                    alpha = 1
                elif neighbor_dst == src:
                    alpha = 1 / random_walk_p
                pq_weight = neighbor_weight * alpha / random_walk_Z
                pq_weights.append(pq_weight)
                new_dst_neighbors.append(neighbor_dst)

            p, a = Node2VecEstimator.setup_alias(pq_weights)

            new_attributes = Row(dst_neighbors=new_dst_neighbors, p=p, a=a)

            return src, dst, new_attributes
        
        
        if self.debug:
            print('Debug - edges:')
            edges.show(10, False)
        
        df = edges.alias('t1').join(edges.alias('t2'), on=(F.col('t1.dst')==F.col('t2.src')), how='inner'). \
                    select('t1.*', \
                           F.col('t2.dst').alias('next_dst'), \
                           F.col('t2.weight').alias('next_weight'))
        if self.debug:
            print('Debug - dst of dst:')
            df.show(10, False)
        
        src_neighbors = edges.groupBy(F.col('src')).agg(F.collect_list(F.col('dst')).alias('src_neighbors'))
        if self.debug:
            print('Debug - src_neighbors:')
            src_neighbors.show(10, False)
        
        df = df.join(src_neighbors, on='src', how='leftouter')
        if self.debug:
            print('Debug - join src_neighbors:')
            df.show(10, False)
        
        df = df.groupBy([F.col('src'), F.col('dst')]).agg(F.struct(F.collect_list(F.struct(F.col('next_dst').alias('dst'),\
                                                                                   F.col('next_weight').alias('weight'))\
                                                                                  ).alias('dst_neighbors'),
                                                                    F.first(F.col('src_neighbors')).alias('src_neighbors')\
                                                                   ).alias('attributes'))
        if self.debug:
            print('Debug - attributes of edges:')
            df.show(10, False)
            df.printSchema()
        
        self.edges_lookup = df.rdd.map(lambda row: _setup_edges(row)).toDF(['src', 'dst', 'attributes'])
        if self.debug:
            print('Debug - edges_lookup:')
            self.edges_lookup.show(10, False)
            self.edges_lookup.printSchema()
    
    def _random_walk(self):
        def _first_step(row):
            src, attributes = row['src'], row['attributes']

            next_index = Node2VecEstimator.draw_alias(attributes['p'], attributes['a'])
            next_vertex = attributes['neighbors'][next_index]

            return src, [src, next_vertice]

        def _next_step(path, attributes):    
            if attributes is not None:
                next_index = Node2VecEstimator.draw_alias(attributes['p'], attributes['a'])
                next_vertex = attributes['dst_neighbors'][next_index]
                path.append(next_vertice)

            return path
        
        walk_df = self.vertices_lookup.rdd.map(lambda row: _first_step(row)).toDF(['origin', 'path'])
        for i in range(self.walk_times-1):
            walk_df = walk_df.union(self.vertices_lookup.rdd.map(lambda row: _first_step(row)).toDF(['origin', 'path']))
                                    
        next_step_udf = udf(lambda path, attributes: _next_step(path, attributes), ArrayType(StringType()))
        for i in range(self.random_walk_steps - 2):
            walk_df = walk_df.withColumn('src', F.element_at(F.col('path'), -2))
            walk_df = walk_df.withColumn('dst', F.element_at(F.col('path'), -1))  
            walk_df = walk_df.join(self.edges_lookup, on=['src', 'dst'], how='leftouter')
            walk_df = walk_df.select('origin', next_step_udf('path', 'attributes').alias('path'))
            
        if self.debug:
            print('Debug - walk_df:')
            walk_df.show(10, False)
            walk_df.printSchema()
            
        return walk_df
    
    def _word2vec(self, random_walk_paths):
        word2Vec = Word2Vec(vectorSize=self.w2v_vector_size, inputCol="path", outputCol="model", \
                            windowSize=self.w2v_window_size, minCount=self.w2v_min_count, \
                            maxIter=self.w2v_max_iter, numPartitions=self.w2v_num_partitions)
        model = word2Vec.fit(random_walk_paths)
        node_vectors = model.getVectors()
        return node_vectors
    
    def _node2vec_transform(self, edges):
        # 1. Initialize lookup dataframe
        self._init_vertices_lookup_df(edges)
        self._init_edges_lookup_df(edges)
        
        # 2. Start random walk
        random_walk_paths = self._random_walk()
        
        # 3. Call Word2Vec
        node_vectors = self._word2vec(random_walk_paths)
        
        return node_vectors
    
    def _get_i2i_df(self, embedding_table):
        
        mh = BucketedRandomProjectionLSH(inputCol='vector', outputCol='hashes', bucketLength=self.euclid_bucket_length)
        model_mh = mh.fit(embedding_table)
        # calculate the distance
        embedding_dist_table = model_mh.approxSimilarityJoin(embedding_table, embedding_table, \
                                                             threshold=self.euclid_distance_threshold, distCol='euclidean_dist')\
                                       .select(F.col('datasetA.word').alias('word_1'),\
                                               F.col('datasetB.word').alias('word_2'),\
                                               F.col('euclidean_dist'))
        
        # distance normalization
        vectorAssembler = VectorAssembler(handleInvalid="keep").setInputCols(['euclidean_dist']).setOutputCol('euclidean_dist_vec')
        embedding_dist_table = vectorAssembler.transform(embedding_dist_table)
        mmScaler = MinMaxScaler(outputCol="scaled_dist").setInputCol("euclidean_dist_vec")
        model = mmScaler.fit(embedding_dist_table)
        embedding_dist_table = model.transform(embedding_dist_table)
        # similarity = 1 - distance
        udf = F.udf(lambda x : float(x[0]), FloatType())
        embedding_sim_table = embedding_dist_table.withColumn('euclidean_sim', 1-udf('scaled_dist'))\
                                                  .drop('euclidean_dist', 'euclidean_dist_vec', 'scaled_dist')\
                                                  .filter(F.col('word_1') != F.col('word_2'))\
                                                  .withColumn('value', F.struct('word_2', 'euclidean_sim'))
        
        # collect the top k list
        max_recommendation_count = self.max_recommendation_count
        w = Window.partitionBy('word_1').orderBy(F.desc('euclidean_sim'))
        recall_df = embedding_sim_table.withColumn('rn',F.row_number()\
                                    .over(w))\
                                    .filter(f'rn <= %d' % max_recommendation_count)\
                                    .groupby('word_1')\
                                    .agg(F.collect_list('value').alias(self.value_column_name))\
                                    .withColumnRenamed('word_1', self.key_column_name)
        return recall_df
    
    def _fit(self, dataset):
        dataset = self._filter_dataset(dataset)
        dataset = self._preprocess_dataset(dataset)
        node_vectors = self._node2vec_transform(dataset)
        df = self._get_i2i_df(node_vectors)
        model = self._create_model(df)
        return model
