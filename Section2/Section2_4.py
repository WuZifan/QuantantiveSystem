import itertools

items=[1,2,3]
# 输出1，2，3三个数不同顺序的所有组合
for item in itertools.permutations(items):
    print(item)

# 输出所有1，2，3三个数的两两组合
for item in itertools.combinations(items,2):
    print(item)

# 输出所有1，2，3三个数的两两组合,数字可以重复
for item in itertools.combinations_with_replacement(items,2):
    print(item)

ab=['a','b']
cd=['c','d']
# 输出的是两者的笛卡尔积
for item in itertools.product(ab,cd):
    print(item)