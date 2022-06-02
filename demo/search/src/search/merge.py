# adapetd from: https://github.com/PaddlePaddle/RocketQA/blob/main/research/DuReader-Retrieval-Baseline/src/merge.py
import sys

#shift = int(sys.argv[1])  # the para index shift for each part
shift = 0

top = int(sys.argv[1])
total_part = int(sys.argv[2])
recall_files = sys.argv[3].split(',')
result_file = sys.argv[4]
assert len(recall_files) <= total_part

f_list = []
for fn in recall_files:
    f0 = open(fn, 'r', encoding='utf8')
    f_list.append(f0)

line_list = []
for f0 in f_list:
    line = f0.readline()
    line_list.append(line)

#out = open('output/dev.res.top%s' % top, 'w')
out = open(result_file, 'w', encoding='utf8')
last_q = ''
ans_list = {}
while line_list[-1]:
    cur_list = []
    for line in line_list:
        sub = line.strip().split('\t')
        cur_list.append(sub)

    if last_q == '':
        last_q = cur_list[0][0]
    if cur_list[0][0] != last_q:
        rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
        for i in range(top):
            out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
        ans_list = {}
    for i, sub in enumerate(cur_list):
        # qid -> rank score
        ans_list[int(sub[1]) + shift*i] = float(sub[-1])
    last_q = cur_list[0][0]

    line_list = []
    for f0 in f_list:
        line = f0.readline()
        line_list.append(line)

rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
for i in range(top):
    out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
out.close()

#print('output/dev.res.top%s' % top)
