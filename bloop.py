import math
import random
import copy

def mat_generate(rows, cols, lower, upper):
    m = []
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(random.uniform(lower, upper))
        m.append(col)

    return m            

def mat_print(x):
    for i in x:
        print(i)

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

    """
    c = []
    for i in range(ra):
        col = []
        for j in range(cb):
            col.append(0)
            for k in range(cb):
                col[j] += a[i][k] * b[k][j]
        c.append(col)
    """

    # Create empty matrix
    c = []
    for i in range(ra):
        col = []
        for j in range(cb):
            col.append(0)
        c.append(col)

    # Multiply matrices
    for i in range(ra):
        for j in range(cb):
            for k in range(ca):
                c[i][j] += a[i][k] * b[k][j]

    return c

def sigmoid(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = 1 / (1 + math.exp(-x[i][j]))

    return x

def forward(train_in, w, b):
    res = mat_dot(train_in, w)
    res = mat_sum(res, b)
    return sigmoid(res)

def cost(train_in, train_out, w, b):
    sum = 0

    res = forward(train_in, w, b)

    for i in range(len(res)):
        for j in range(len(res[i])):
            sum += (res[i][j] - train_out[i][j]) ** 2

    return sum / (len(res) # * len(res[0]))

def derivative_w(train_in, train_out, i, j, w, b, h):
    w_inc = copy.deepcopy(w)
    w_inc[i][j] += h
    dw = (cost(train_in, train_out, w_inc, b) - cost(train_in, train_out, w, b)) / h

    return dw

def derivative_b(train_in, train_out, i, j, w, b, h):
    b_inc = copy.deepcopy(b)
    b_inc[i][j] += h
    db = (cost(train_in, train_out, w, b_inc) - cost(train_in, train_out, w, b)) / h

    return db

def train(count, train_in, train_out, w, b, h, rate):
    for x in range(count):
        dw = mat_generate(len(w), len(w[0]), 0, 0)
        db = mat_generate(len(b), len(b[0]), 0, 0)

        # Differentiate weights and store in dw
        for i in range(len(w)):
            for j in range(len(w[i])):
                dw[i][j] = derivative_w(train_in, train_out, i, j, w, b, h)

        # Differentiate biases and store in db
        for i in range(len(b)):
            for j in range(len(b[i])):
                db[i][j] = derivative_b(train_in, train_out, i, j, w, b, h)

        # Update weights
        for i in range(len(w)):
            for j in range(len(w[i])):
                w[i][j] -= dw[i][j] * rate

        # Update biases
        for i in range(len(b)):
            for j in range(len(b[i])):
                b[i][j] -= db[i][j] * rate

        # Printing
        """
        res = mat_dot(train_in, w)
        res = mat_sum(res, b)
        res = sigmoid(res)
        print("\nRES")
        mat_print(res)
        print("\nW")
        mat_print(w)
        print("\nB")
        mat_print(b)
        """

    return w, b

def main():
    train_in = [[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]]

    train_out = [[0],
                 [1],
                 [1],
                 [0]]

    h = 1e-3
    rate = 1e-0

    # 4x2 2x1 => 4x1

    w = mat_generate(2, 1, 0, 1)
    b = mat_generate(len(train_in), len(w[0]), 0, 1)
    # y = x * w + b

    res = forward(train_in, w, b)
    mat_print(res)
    print()

    w, b = train(1000, train_in, train_out, w, b, h, rate)

    print()
    res = forward(train_in, w, b)
    mat_print(res)

    # Create two layers: (First layer -> OR NAND gates, Second layer -> AND) -> XOR
    # Combine first layer into matrix for second layer
    # Need three weight matrices and bias matrices


if __name__ == "__main__":
    main()
