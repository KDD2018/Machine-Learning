

class Node(object):
    """节点"""
    def __init__(self, elem):
        self.elem = elem
        self.next = None
        self.prior = None


class DoubleLinkList(object):
    """双向链表"""
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
        self.__head.prior = node

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
            node.prior = cur

    def insert(self, pos, item):
        """指定位置插入元素"""
        if pos <= 0:
            self.add(item)
        elif pos > self.length-1:
            self.append(item)
        else:
            cur = self.__head
            count = 0
            while count < pos:
                count += 1
                cur = cur.next
            # 退出循环时，prior指向pos
            node = Node(item)
            node.next = cur
            node.prior = cur.prior
            cur.prior = node
            node.prior.next = node

    def remove(self, item):
        """删除指定元素"""
        cur = self.__head
        while cur is not None:
            if cur.elem == item:
                if cur == self.__head:
                    # 头结点
                    self.__head = cur.next
                    if cur.next:
                        # 判断链表是否只有一个节点
                        cur.next.prior = None
                else:
                    cur.prior.next = cur.next
                    if cur.next:
                        cur.next.prior = cur.prev
                break
            else:
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
    dl = DoubleLinkList(Node(10))
    print(dl.is_empty)
    print(dl.length)

    dl.add(100)
    print(dl.is_empty)
    print(dl.length)
    dl.travel()

    dl.append(3)
    dl.append(5)
    dl.add(0)
    dl.travel()

    dl.insert(2, 999)
    dl.travel()
    dl.insert(6, 77)
    dl.travel()
    dl.insert(-2, 666)
    dl.travel()

    dl.remove(0)
    dl.travel()
    print(dl.search(0))
    print(dl.search(5))