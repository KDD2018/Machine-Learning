class Queue(object):
    """队列---先进先出"""
    def __init__(self):
        self.__list = []

    def enqueue(self, item):
        """入队，添加新元素到队尾"""
        self.__list.append(item)

    def dequeue(self):
        """出队， 从队首删除元素"""
        return self.__list.pop(0)

    def is_empty(self):
        """判断栈是否为空"""
        return self.__list == []

    def size(self):
        """返回栈的元素个数"""
        return len(self.__list)


if __name__ == '__main__':
    queue = Queue()
    queue.enqueue(1)
    queue.enqueue(3)
    queue.enqueue(5)
    queue.enqueue(7)
    print(queue.dequeue())
    print(queue.dequeue())
    print(queue.dequeue())




