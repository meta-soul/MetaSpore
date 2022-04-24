import sys

res = {}
for line in sys.stdin:
    line = line.strip('\r\n')
    if not line:
        continue
    model, exp, score = line.split('\t')
    if model not in res:
        res[model] = []
    res[model].append((exp, score))

cols = [e for e,s in list(res.items())[0][1]]
print('', *cols, sep='\t')
for model, exp_list in res.items():
    exp_list = sorted(exp_list, key=lambda x:cols.index(x[0]))
    print(model, *[x[1] for x in exp_list], sep='\t')
