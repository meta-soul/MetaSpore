from elasticsearch import Elasticsearch

def create_es_using_http_auth(http_uri, user, password, verify_certs=False, **params):
    es = Elasticsearch([http_uri], http_auth=(user, password), verify_certs=verify_certs)
    es.info(pretty=True)
    return es

def create_index(es, index_name, index_schema, **params):
    result = es.indices.create(index=index_name, body=index_schema)
    return es

def delete_index(es, index_name, **params):
    if es.indices.exists(index=index_name):
        result = es.indices.delete(index=index_name)
    return es

def save_df_to_es_index(es, df, index_name, mapping_id=None, **params):
    if mapping_id is not None:
        df.write\
          .format("org.elasticsearch.spark.sql")\
          .option("es.mapping.id", mapping_id) \
          .mode("overwrite") \
          .save(index_name)
    else:
        df.write\
          .format("org.elasticsearch.spark.sql")\
          .mode("overwrite") \
          .save(index_name)
    return es

def generate_keyword_match_rules(match_rules, using_phrase=True):
    if match_rules is None:
        return None
    rules = []
    match_method = 'match_phrase' if using_phrase else 'match'
    for field, match_keyword in match_rules:
        match_rule = {match_method: {field: match_keyword}}
        rules.append(match_rule)
    return rules

def generate_attribute_sorter_rules(sorter_rules):
    if sorter_rules is None: 
        return None
    rules = []
    for attr, order in sorter_rules:
        sorter_rule = {attr: {"order": order}}
        rules.append(sorter_rule)
    return rules

def generte_id_rules(ids):
    return {"ids": {"values": [str(x) for x in ids]}} if ids else None

def generate_searh_bool_query(must_rules,
                              should_rules,
                              must_not_rules,
                              filter_rules):
    bool_query = ({
        "bool": {
            "must": must_rules,
            "should": should_rules,
            "must_not": must_not_rules,
            "filter": filter_rules
    }})
    return bool_query

def search_es_using_query_combination(es, 
                                      index_name,
                                      must_rules=None,
                                      should_rules=None,
                                      must_not_rules=None,
                                      filter_rules=None,
                                      sorter_rules=None,
                                      from_no=None,
                                      size=None,
                                      **params):
    bool_query = generate_searh_bool_query(must_rules, 
                                           should_rules, 
                                           must_not_rules, 
                                           filter_rules)
    result = es.search(
                index=index_name, 
                query=bool_query, 
                sort=sorter_rules,
                from_=from_no,
                size=size)
    return result

def search_es_using_id_filtering(es, 
                                 index_name,
                                 id_rules=None,
                                 filter_rules=None,
                                 sorter_rules=None,
                                 from_no=None,
                                 size=None,
                                 **params):
    result = es.search(
                index=index_name, 
                query=id_rules, 
                sort=sorter_rules,
                from_=from_no,
                size=size)
    return result

def parse_es_search_result(all_hits):
    result_list=[hit['_source'] for hit in all_hits['hits']['hits']]
    return result_list