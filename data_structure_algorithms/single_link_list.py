
class Node(object):
    """节点"""
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    """单向链表"""
    def __init__(self, node=None):
        self.__head = node

    @property
    def is_empty(self):
        """链表是否为空"""
        return self.__head is None

    @property
    def length(self):
        """链表长度"""
        # 定义游标，用于遍历链表中的节点
        cur = self.__head
        # 初始化节点数
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """遍历链表节点"""
        cur = self.__head
        while cur is not None:
            print(cur.elem, end=" ")
            cur = cur.next
        print("")

    def add(self, item):
        """链表头部添加元素"""
        node = Node(item)
        node.next = self.__head
        self.__head = node

    def append(self, item):
        """链表尾部添加元素"""
        node = Node(item)
        if self.is_empty:
            self.__head = node
        else:
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
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
        cur = self.__head
        prior = None
        while cur is not None:
            if cur.elem == item:
                if cur == self.__head:
                    self.__head = cur.next
                else:
                    prior.next = cur.next
                break
            else:
                prior = cur
                cur = cur.next

    def search(self, item):
        """查找是否存在某元素"""
        cur = self.__head
        while cur is not None:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        return False


if __name__ == '__main__':
    sl = SingleLinkList(Node(10))
    print(sl.is_empty)
    print(sl.length)

    sl.add(100)
    print(sl.is_empty)
    print(sl.length)
    sl.travel()

    sl.append(3)
    sl.append(5)
    sl.add(0)
    sl.travel()

    sl.insert(2, 999)
    sl.travel()
    sl.insert(6, 77)
    sl.travel()
    sl.insert(-2, 666)
    sl.travel()

    sl.remove(0)
    sl.travel()
    print(sl.search(0))
    print(sl.search(5))
