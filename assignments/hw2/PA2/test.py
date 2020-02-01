class B(object):
    """docstring for B"""

    def __init__(self, arg):
        super(B, self).__init__()
        self.arg = arg
        self.c = [1, 2]


class A(object):
    """docstring for A"""

    def __init__(self, arg):
        super(A, self).__init__()
        self.arg = arg
        self.b = [B(arg), B(arg + 1)]


a1 = A(1)
c1 = a1.b[0].c.copy()
a1.b[0].c[0] = 3
print(c1)
