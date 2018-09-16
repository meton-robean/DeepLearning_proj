import collections

label  = [0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 1, 1, 0, 2, 2, 2, 0]
result = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

total_num = len(label)
cluster_counter = collections.Counter(result)
orginal_counter = collections.Counter(label)

for k in cluster_counter:
    print (k)
for j in orginal_counter:
    print (j)