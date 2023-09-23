import math
import random
import copy

train_in = [[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]]

train_out = [[0],
             [1],
             [1],
             [0]]

def mat_sum(a, b):
    ra = len(a)
    ca = len(a[0])
    rb = len(b)
    cb = len(b[0])
    assert ra == rb
    assert ca == cb

    for i in range(ra):
        for j in range(ca):
            a[i][j] += b[i][j]

    return a


def mat_dot(a, b):
    ra = len(a)
    ca = len(a[0])
    rb = len(b)
    cb = len(b[0])
    assert ca == rb

    # UNSTABLE MFER
    c = []
    for i in range(ra):
        col = []
        for j in range(cb):
            col.append(0)
            for k in range(ca):
                col[j] += a[i][k] * b[k][j]
        c.append(col)

    return c


def mat_insert_col(dest, col, index):
    rd = len(dest)
    cd = len(dest[0])
    rc = len(col)
    cc = len(col[0])
    assert rd == rc
    assert cc == 1
    assert index < cd

    for i in range(rd):
        dest[i][index] = col[i][0]


def mat_print(x):
    for i in x:
        print(i)


def mat_gen(rows, cols, lower, upper):
    m = []
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(random.uniform(lower, upper))
        m.append(col)

    return m            


class Network:
    def __init__(self, arch: list):
        self._arch = arch
        self._ws = []
        self._bs = []
        self._as = []

        rows = len(train_in)

        self._as.append(mat_gen(rows, len(train_in[0]), 0, 0))
        for i in range(len(self._arch)):
            for j in range(self._arch[i]):
                self._ws.append(mat_gen(len(self._as[i][0]), 1, 0, 1))
                self._bs.append(mat_gen(rows, 1, 0, 1))
            self._as.append(mat_gen(rows, self._arch[i], 0, 0))


    def sigmoid(self, x):
        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] = 1 / (1 + math.exp(-x[i][j]))


    def forward(self):
        # For each layer
            # For each neuron
                # Multiply by w and add b
                # Activate
        n_count = 0
        for i in range(len(self._arch)):
            for j in range(self._arch[i]):
                res = mat_dot(self._as[i], self._ws[n_count])
                res = mat_sum(res, self._bs[n_count])
                self.sigmoid(res)
                mat_insert_col(self._as[i + 1], res, j)
                n_count += 1


    def out(self):
        return self._as[len(self._arch)]


    def print(self):
        for i in range(len(self._as)):
            print("\nas")
            mat_print(self._as[i])
                        
        for i in range(len(self._ws)):
            print("\nw:")
            mat_print(self._ws[i])    
            print("\nb:")
            mat_print(self._bs[i])


def main():
    nn = Network([2, 1])
    nn.print()
    nn.forward()
    nn.print()
    print()
    mat_print(nn.out())
    

if __name__ == "__main__":
    main()
