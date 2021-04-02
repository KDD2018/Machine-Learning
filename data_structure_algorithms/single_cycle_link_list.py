
class Node(object):
    """节点"""
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleCycleLinkList(object):
    """单向循环链表"""
    def __init__(self, node=None):
        self.__head = node
        if node:
            node.next = node

    @property
    def is_empty(self):
        """链表是否为空"""
        return self.__head is None

    @property
    def length(self):
        """链表长度"""
        if self.is_empty:
            return 0
        # 定义游标，用于遍历链表中的节点
        cur = self.__head
        # 初始化节点数
        count = 1
        while cur.next is not self.__head:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """遍历链表节点"""
        if self.is_empty:
            return
        cur = self.__head
        while cur.next is not self.__head:
            print(cur.elem, end=" ")
            cur = cur.next
            # 退出循环时，cur指向尾节点
        print(cur.elem)

    def add(self, item):
        """链表头部添加元素"""
        node = Node(item)
        if self.is_empty:
            self.__head = node
            node.next = node
        else:
            cur = self.__head
            while cur.next != self.__head:
                cur = cur.next
            # 退出循环，cur指向尾节点
            node.next = self.__head
            self.__head = node
            cur.next = self.__head

    def append(self, item):
        """链表尾部添加元素"""
        node = Node(item)
        if self.is_empty:
            self.__head = node
            node.next = node
        else:
            cur = self.__head
            while cur.next is not self.__head:
                cur = cur.next
            node.next = self.__head
            cur.next = node

    def insert(self, pos, item):
        """指定位置插入元素"""
        if pos <= 0:
            self.add(item)
        elif pos > self.length-1:
            self.append(item)
        else:
            prior = self.__head
            count = 1
            while count < pos:
                count += 1
                prior = prior.next
            # 退出循环时，prior指向pos-1
            node = Node(item)
            node.next = prior.next
            prior.next = node

    def remove(self, item):
        """删除指定元素"""
        if self.is_empty:
            return
        cur = self.__head
        prior = None
        while cur.next is not self.__head:
            if cur.elem == item:
                if cur == self.__head:
                    # 头节点
                    rear = self.__head
                    while rear.next is not self.__head:
                        rear = rear.next
                    self.__head = cur.next
                    rear.next = self.__head
                else:
                    prior.next = cur.next
                return
            else:
                prior = cur
                cur = cur.next
        # 退出循环时，cur指向尾节点
        if cur.elem == item:
            if cur == self.__head:
                self.__head = None
            else:
                prior.next = self.__head

    def search(self, item):
        """查找是否存在某元素"""
        if self.is_empty:
            return False
        cur = self.__head
        while cur.next is not self.__head:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        # 退出循环时，cur指向尾节点
        if cur.elem == item:
            return True
        return False


if __name__ == '__main__':
    sl = SingleCycleLinkList(Node(10))  # 10
    print(sl.is_empty)
    print(sl.length)

    sl.add(100)  # 100 10
    print(sl.is_empty)
    print(sl.length)
    sl.travel()

    sl.append(3)  # 100 10 3
    sl.append(5)
    sl.add(0)  # 0 100 10 3 5
    sl.travel()
    print(sl.length)

    sl.insert(2, 999)  # 0 100 999 10 3 5
    sl.travel()
    sl.insert(6, 77)  # 0 100 999 10 3 5 77
    sl.travel()
    sl.insert(-2, 666)  # 666 0 100 999 10 3 5 77
    sl.travel()

    sl.remove(0)
    sl.travel()
    print(sl.search(0))
    print(sl.search(5))
