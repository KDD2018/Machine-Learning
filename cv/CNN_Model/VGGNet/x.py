import math


def solve(n, m, fc1, fc2, x):
    w1 = [[0 for _ in range(n)] for _ in range(m)]
    w2 = [[0 for _ in range(m)] for _ in range(10)]
    for i in range(m):
        for j in range(n):
            w1[i][j] = fc1[i * n + j]
    for i in range(10):
        for j in range(m):
            w2[i][j] = fc2[i * m + j]
    w1x = [0] * m
    for i in range(m):
        for j in range(n):
            w1x[i] += w1[i][j] * x[j]
    w1x_relu = [0] * m
    for i in range(m):
        w1x_relu[i] = max(0, w1x[i])

    w2x = [0] * 10
    for i in range(10):
        for j in range(m):
            w2x[i] += w2[i][j] * w1x_relu[j]
    z_max = max(w2x)
    org_class = w2x.index(z_max)
    w2x_exp = [0] * 10
    for i in range(10):
        w2x_exp[i] = math.exp(w2x[i] - z_max)
    z_sum = sum(w2x_exp)
    w2x_softmax = [0] * 10
    for i in range(10):
        w2x_softmax[i] = w2x_exp[i] / z_sum
    probilities = [0] * 10
    probilities[org_class] = w2x_softmax[org_class]

    tmp_w1x_relu = [0] * m
    tmp_w2x = [0] * 10
    tmp_w2x_exp = [0] * 10
    tmp_w2x_softmax = [0] * 10
    pv = [None] * 10
    for i in range(n):
        tmp_w1x = w1x[:]
        delta = -129 - x[i]
        for k in range(m):
            tmp_w1x[k] += delta * w1[k][i]
        for val in range(-128, 128):
            for k in range(m):
                tmp_w1x[k] += w1[k][i]
            for k in range(m):
                tmp_w1x_relu[k] = max(tmp_w1x[k], 0)
            for i_ in range(10):
                tmp_w2x[i_] = 0
                for j_ in range(m):
                    tmp_w2x[i_] += w2[i_][j_] * tmp_w1x_relu[j_]
            tmp_z_max = max(tmp_w2x)
            tmp_class = tmp_w2x.index(tmp_z_max)
            for k in range(10):
                tmp_w2x_exp[k] = math.exp(tmp_w2x[k] - tmp_z_max)
            tmp_z_sum = sum(tmp_w2x_exp)
            for k in range(10):
                tmp_w2x_softmax[k] = tmp_w2x_exp[k] / tmp_z_sum
            if tmp_class == org_class:
                if tmp_w2x_softmax[tmp_class] <= probilities[tmp_class]:
                    probilities[tmp_class] = tmp_w2x_softmax[tmp_class]
                    pv[tmp_class] = (i + 1, val)

            else:
                if tmp_w2x_softmax[tmp_class] >= probilities[tmp_class]:
                    probilities[tmp_class] = tmp_w2x_softmax[tmp_class]
                    pv[tmp_class] = (i + 1, val)
    cnt = 0
    for idx, val in enumerate(pv):
        if val is not None:
            cnt += 1
    if cnt == 1:
        print('{} {}'.format(*pv[org_class]))
    else:
        ret = 0
        prep = 0
        for idx, val in enumerate(probilities):
            if idx == org_class:
                continue
            if val > prep:
                prep = val
                ret = idx
        print('{} {}'.format(*pv[ret]))


if __name__ == '__main__':
    N, M = map(int, input().split())
    X = list(map(int, input().split()))
    lc1 = list(map(float, input().split()))
    lc2 = list(map(float, input().split()))
    solve(N, M, lc1, lc2, X)
