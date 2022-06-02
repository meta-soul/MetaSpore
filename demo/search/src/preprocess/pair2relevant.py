import os
import sys

# input
pair_file = sys.argv[1]  # pos pair without label for now
# output
query_file = sys.argv[2]
para_file = sys.argv[3]
rel_file = sys.argv[4]

qid_map = {}
pid_map = {}
rel_map = {}
with open(pair_file, 'r', encoding='utf8') as fin:
    for line in fin:
        query, qid, para, pid = line.strip().split('\t')
        qid_map[qid] = query
        pid_map[pid] = para
        if qid not in rel_map:
            rel_map[qid] = set()
        rel_map[qid].add(pid)

rel_n = sum([len(s) for k, s in rel_map.items()])
print(f"Queries: {len(qid_map)}")
print(f"Passages: {len(pid_map)}")
print(f"Relevant: {rel_n}")

with open(query_file, 'w', encoding='utf8') as f:
    for qid, query in qid_map.items():
        print(qid, query, sep='\t', file=f)

with open(para_file, 'w', encoding='utf8') as f:
    for pid, para in pid_map.items():
        print(pid, para, sep='\t', file=f)

with open(rel_file, 'w', encoding='utf8') as f:
    for qid, rels in rel_map.items():
        rels = list(rels)
        if not rels:
            continue
        print(qid, ','.join(rels), sep='\t', file=f)
