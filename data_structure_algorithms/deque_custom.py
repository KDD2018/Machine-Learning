class Deque(object):
    """双端队列---先进先出"""
    def __init__(self):
        self.__list = []

    def add_front(self, item):
        """往队首添加元素"""
        self.__list.insert(0, item)

    def add_rear(self, item):
        """添加新元素到队尾"""
        self.__list.append(item)

    def pop_front(self):
        """从队首删除元素"""
        return self.__list.pop(0)

    def pop_rear(self):
        """从队尾删除元素"""
        return self.__list.pop()

    def is_empty(self):
        """判断栈是否为空"""
        return self.__list == []

    def size(self):
        """返回栈的元素个数"""
        return len(self.__list)


if __name__ == '__main__':
    deque = Deque()
    deque.add_rear(1)
    deque.add_rear(3)
    deque.add_rear(5)
    deque.add_rear(7)
    deque.add_front(99)
    deque.add_front(88)
    print(deque.pop_front())
    print(deque.pop_front())
    print(deque.pop_front())
    print(deque.pop_rear())




