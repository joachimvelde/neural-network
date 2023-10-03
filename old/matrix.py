# This file contains various matrix operations.
# They are slower than numpy matrices, but less cringe

import random

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

