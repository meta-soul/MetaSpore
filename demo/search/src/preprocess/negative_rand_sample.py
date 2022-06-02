import sys
import random


def sample(query, queries, neg_ratio=5):
    neg_list = []
    num_neg = len(queries[query]) * neg_ratio
    data = list(queries.items())
    for i in range(num_neg):
        # rand choice a neg from one random query
        while True:
            neg_query, neg_cands = random.choice(data)
            if neg_query != query and len(neg_cands) > 0:
                break
        neg = random.choice(neg_cands)
        neg_list.append(neg)
    return neg_list


if __name__ == '__main__':
    neg_ratio = int(sys.argv[1])
    output_fmt = sys.argv[2]
    assert output_fmt in ['pair', 'triplet']

    # load data
    queries = {}
    for line in sys.stdin:
        query, pos = line.strip().split('\t')
        if query not in queries:
            queries[query] = []
        queries[query].append(pos)

    # sample and dump
    for query in queries:
        pos_list = queries[query]
        neg_list = sample(query, queries, neg_ratio=neg_ratio)
        if output_fmt == 'pair':
            for pos in pos_list:
                print(query, pos, 1, sep='\t')
            for neg in neg_list:
                print(query, neg, 0, sep='\t')
        elif output_fmt == 'triplet':
            for pos in pos_list:
                for neg in neg_list:
                    print(query, pos, neg, sep='\t')
