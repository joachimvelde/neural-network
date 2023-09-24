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


def mat_sub(a, b):
    ra = len(a)
    ca = len(a[0])
    rb = len(b)
    cb = len(b[0])
    assert ra == rb
    assert ca == cb

    for i in range(ra):
        for j in range(ca):
            a[i][j] -= b[i][j]

    return a


def mat_dot(a, b):
    ra = len(a)
    ca = len(a[0])
    rb = len(b)
    cb = len(b[0])
    assert ca == rb

    c = []
    for i in range(ra):
        col = []
        for j in range(cb):
            col.append(0)
            for k in range(ca):
                col[j] += a[i][k] * b[k][j]
        c.append(col)

    return c


def mat_scalar_dot(mat, scalar):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] *= scalar


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

        self._h = 1e-3
        self._rate = 1e-1

        # Temporary for bad gradient calculation
        self._wd = []
        self._bd = []

        rows = len(train_in)

        self._as.append(mat_gen(rows, len(train_in[0]), 0, 0))
        # Insert train in into first as
        mat_sum(self._as[0], train_in)

        for i in range(len(self._arch)):
            for j in range(self._arch[i]):
                self._ws.append(mat_gen(len(self._as[i][0]), 1, 0, 1))
                self._bs.append(mat_gen(rows, 1, 0, 1))

                # Bad gradient variables
                self._wd.append(mat_gen(len(self._as[i][0]), 1, 0, 0))
                self._bd.append(mat_gen(rows, 1, 0, 0))

            self._as.append(mat_gen(rows, self._arch[i], 0, 0))


    def sigmoid(self, x):
        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] = 1 / (1 + math.exp(-x[i][j]))


    def forward(self):
        n_count = 0
        for i in range(len(self._arch)):
            for j in range(self._arch[i]):
                res = mat_dot(self._as[i], self._ws[n_count])
                res = mat_sum(res, self._bs[n_count])
                self.sigmoid(res)
                mat_insert_col(self._as[i + 1], res, j)
                n_count += 1


    def cost(self):
        sum = 0
        self.forward()
        output = self.out()
        for i in range(len(output)):
            d = (train_out[i][0] - output[i][0])
            sum += d**2
        return sum / len(output)


    def train(self, count, load=False):
        for x in range(count):
            # Differentiate weights and biases - store in wd and wb
            n_count = 0
            for i in range(len(self._arch)):
                for j in range(self._arch[i]):
                    self.diff_w(n_count)
                    self.diff_b(n_count) 
                    n_count += 1

            # Update weights and biases using wd and wb
            n_count = 0
            for i in range(len(self._arch)):
                for j in range(self._arch[i]):
                    self._ws[n_count] = mat_sub(self._ws[n_count], self._wd[n_count])
                    self._bs[n_count] = mat_sub(self._bs[n_count], self._bd[n_count])
                    n_count += 1

            # Custom made by Audun
            if x % (count / 100) == 0 and load:
                loading_count = x / count
                if loading_count * 100 % 1 == 0:
                    print("Loading... {0}%".format(loading_count * 100))


    def diff_w(self, index):
        w = self._ws[index]
        wd = self._wd[index]
        for i in range(len(w)):
            for j in range(len(w[i])):
                w[i][j] += self._h
                f_inc = self.cost()
                w[i][j] -= self._h
                f = self.cost()
                wd[i][j] = (f_inc - f) / self._h


    def diff_b(self, index):
        b = self._bs[index]
        bd = self._bd[index]
        for i in range(len(b)):
            for j in range(len(b[i])):
                b[i][j] += self._h 
                f_inc = self.cost()
                b[i][j] -= self._h
                f = self.cost()
                bd[i][j] = (f_inc - f) / self._h


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
    nn.train(10000)
    mat_print(nn.out())


if __name__ == "__main__":
    main()
