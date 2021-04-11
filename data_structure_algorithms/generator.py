def yield_test(n):
    for i in range(n):
        print('******1******')
        yield call(i)
        print('******2******')
        print("i=",i)
    print("Done.")

def call(i):
    return i*2

for i in yield_test(5):
    print(i,",")