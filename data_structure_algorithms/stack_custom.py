class Stack(object):
    """栈---后进先出"""
    def __init__(self):
        self.__list = []

    def push(self, item):
        """入栈，添加新元素到栈顶"""
        self.__list.append(item)

    def pop(self):
        """出栈， 弹出栈顶元素"""
        return self.__list.pop()

    def peek(self):
        """返回栈顶元素"""
        if self.__list:
            return self.__list[-1]
        else:
            return None

    def is_empty(self):
        """判断栈是否为空"""
        return self.__list == []

    def size(self):
        """返回栈的元素个数"""
        return len(self.__list)


if __name__ == '__main__':
    stack = Stack()
    stack.push(1)
    stack.push(3)
    stack.push(5)
    stack.push(7)
    print(stack.pop())
    print(stack.pop())
    print(stack.pop())




