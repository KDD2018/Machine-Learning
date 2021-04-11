class Fibonacci(object):
    def __init__(self, num):
        self.num = num
        self.a = 0
        self.b = 1
        self.cur_num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_num < self.num:
            ret = self.a
            self.a, self.b = self.b, self.a + self.b
            self.cur_num += 1
            return ret
        else:
            raise StopIteration


if __name__ == '__main__':

    Fibonacci = Fibonacci(10)
    for num in Fibonacci:
        print(num)


