import time
from collections.abc import Iterable, Iterator


class ClassIterator(object):
    def __init__(self, obj):
        self.obj = obj
        self.cur_num = 0

    def __iter__(self):
        pass

    def __next__(self):
        if self.cur_num < len(self.obj.names):
            ret = self.obj.names[self.cur_num]
            self.cur_num += 1
            return ret
        else:
            raise StopIteration


class Classmate(object):
    def __init__(self):
        self.names = list()

    def add(self, name):
        self.names.append(name)

    def __iter__(self):
        return ClassIterator(self)


if __name__ == '__main__':

    classmate = Classmate()
    classmate.add('张三')
    classmate.add('李四')
    classmate.add('王五')
    classmate.add('麻二')
    print('是否是可迭代对象：', isinstance(classmate, Iterable))
    print('是否是迭代器：', isinstance(classmate, Iterator))

    for name in classmate:
        time.sleep(1)
        print(name)


