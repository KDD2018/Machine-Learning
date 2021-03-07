
def f(mid):
    global las
    global fir
    root = fir[0]
    fir = fir[1:]
    root_po = mid.find(root)
    left = mid[0:root_po]
    right = mid[root_po+1:len(mid)]
    if len(left) >0:
        f(left)
    if len(right) > 0:
        f(right)
    las += root

if __name__ == '__main__':
    fir = input()
    mid = input()
    las = ''
    f(mid)
    print(las)