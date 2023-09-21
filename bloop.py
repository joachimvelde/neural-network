import math
import random

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

def cost(train_in, train_out, w, b):
    sum = 0

    res = mat_dot(train_in, w)
    res = mat_sum(res, b)
    res = sigmoid(res)

    for i in range(len(res)):
        for j in range(len(res[i])):
            sum += (res[i][j] - train_out[i][j]) ** 2

    return sum / (len(res) * len(res[0]))

def derivative_w(train_in, train_out, i, j, w, b, h):
    w_inc = w
    w_inc[i][j] += h
    dw = (cost(train_in, train_out, w_inc, b) - cost(train_in, train_out, w, b)) / h

    return dw

def derivative_b(train_in, train_out, i, j, w, b, h):
    b_inc = b
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
        res = mat_dot(train_in, w)
        res = mat_sum(res, b)
        res = sigmoid(res)
        print("\nRES")
        mat_print(res)
        print("\nW")
        mat_print(w)
        print("\nB")
        mat_print(b)

    return w, b

def main():
    train_in = [[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]]

    train_out = [[0],
                 [0],
                 [0],
                 [1]]

    h = 1e-3
    rate = 1e-3

    # 4x2 2x1 => 4x1

    w = mat_generate(2, 1, 0, 1)
    b = mat_generate(len(train_in), len(w[0]), 0, 1)
    # y = x * w + b

    res = mat_dot(train_in, w)
    res = mat_sum(res, b)
    res = sigmoid(res)
    mat_print(res)
    print()

    w, b = train(10000, train_in, train_out, w, b, h, rate)

    print()
    res = mat_dot(train_in, w)
    res = mat_sum(res, b)
    res = sigmoid(res)
    mat_print(res)

    # Implement biases
    # apply sigmoid

if __name__ == "__main__":
    main()

"""
def cost(w1, w2, b):
    sum = 0
    for i in range(len(train_in)):
        res = train_in[i][0] * w1 + train_in[i][1] * w2 + b
        diff = (res - train_out[i][0])**2
        sum += diff
    return sum / len(train_in)

def train(count, w1, w2, b):
    for i in range(count):
        dw1 = (cost(w1 + h, w2, b) - cost(w1, w2, b)) / h
        dw2 = (cost(w1, w2 + h, b) - cost(w1, w2, b)) / h
        db  = (cost(w1, w2, b + h) - cost(w1, w2, b)) / h
        w1 -= dw1 * rate
        w2 -= dw2 * rate
        b  -= db  * rate
        # print_res(w1, w2, b)
    return w1, w2, b
"""
