import numpy as np
import io


class act_t(int):
    @staticmethod
    def test():
        a = act_t(1)
        print("act_t(1):", a)


class Node(object):
    MEM = {}
    N = 16

    def __init__(self):
        self.parent = NodePtr(0)
        self.access = np.empty(shape=[2], dtype=act_t)
        self.actA = np.empty(shape=[0], dtype=act_t)
        self.actB = np.empty(shape=[0], dtype=act_t)
        self.child = np.empty(shape=[0, 0], dtype=NodePtr)
        self.ptr = NodePtr(0)

    def __repr__(self):
        buff = io.StringIO()
        print("{", file=buff)
        print("h:", self.h(), file=buff)
        print("actA:", self.actA, file=buff)
        print("actB:", self.actB, file=buff)
        print("}", end="", file=buff)
        return buff.getvalue()

    def __str__(self):
        return self.__repr__()

    def h(self):
        p = self.ptr
        y = np.empty([0, 2], dtype=act_t)
        while p.o.parent != NodePtr(0):
            y = np.append(y, p.o.access[None, :], axis=0)
            p = p.o.parent
        y = np.flip(y, axis=0)
        y = y.tolist()
        return y

    def __call__(self, acts) -> "Node":
        p = self.ptr
        for ab in acts:
            a, b = ab
            i = np.where(p.o.actA == a)
            j = np.where(p.o.actB == b)
            if len(i[0]) and len(j[0]):
                p = p.o.child[i[0], j[0]]
            else:
                p = self.ptr
                print("Warning: invalid action sequence, nothing will take effect.")
                break
        return p.o

    @classmethod
    def new(cls) -> "NodePtr":
        if len(cls.MEM) >= 0.75 * cls.N:
            cls.N *= 2
        while True:
            p = np.random.randint(1, cls.N)
            if p not in cls.MEM.keys():
                break
        p = NodePtr(p)
        p.o = Node()
        p.o.ptr = p
        return p

    @classmethod
    def news(cls, qty: int) -> "NodePtr":
        pass

    def get_child(self, i, j) -> "NodePtr":
        c: NodePtr = self.child[i, j]
        return c

    @classmethod
    def test(cls):
        root = cls.new()
        root.o.access = np.array([0, 0])
        root.o.actA = np.array([1, 2, 3])
        root.o.actB = np.array([1, 2, 3])
        root.o.child = np.resize(root.o.child, [3, 3])
        print(root.o)
        print("child.shape:", root.o.child.shape)


class NodePtr(int):
    @property
    def o(self) -> Node:
        return Node.MEM[self]

    @o.getter
    def o(self) -> Node:
        return Node.MEM[self]

    @o.setter
    def o(self, value: "Node"):
        Node.MEM[self] = value

    def __repr__(self):
        buff = io.StringIO()
        print(self.o.h(), end="", file=buff)
        return buff.getvalue()


if __name__ == "__main__":
    act_t.test()
    Node.test()
    n = Node.new()
    print(n)
