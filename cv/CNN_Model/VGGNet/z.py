import math


def upsample1(d, p):
    # 普通结界
    assert 1 <= p <= 10
    return d + p


def upsample2(d, p):
    # 倍增结界
    assert 2 <= p <= 3
    return d * p


def downsample(d, p):
    # 聚集结界
    assert 2 <= p <= 10
    return math.ceil(d / p)


# 初始化杀伤力范围
lethal_radius = 1

# 结界参数(z, p)
config = [(1, 6),
          (2, 3),
          (3, 3),
          (2, 3),
          (2, 3),
          (3, 7)]

for i in range(int(input())):
    z, p = list(map(int, input().strip().split()))
    if z == 1:
        lethal_radius = upsample1(lethal_radius, p)
    if z == 2:
        lethal_radius = upsample2(lethal_radius, p)
    if z == 3:
        lethal_radius = downsample(lethal_radius, p)
print(lethal_radius)



