import math
import random

train_in = [0,
            1,
            2,
            3,
            4,
            5]

train_out = [0,
             2,
             4,
             6,
             8,
             10]

h = 1e-10
rate = 1e-1

def cost(w, b):
    sum = 0
    for i in range(len(train_in)):
        res = train_in[i] * w + b
        diff = (train_out[i] - res)**2
        sum += diff
    return sum / len(train_in)

def train(count, w, b):
    for i in range(count):
        print(f"w: {w}")
        print(f"cost: {cost(w, b)}\n")
        # print_res(w)
        dw = (cost(w + h, b) - cost(w, b)) / h
        db = (cost(w, b + h) - cost(w, b)) / h
        w -= dw * rate
        b -= db * rate
    return w, b

def print_res(w, b):
    for i in range(len(train_in)):
        res = "{:.11f}".format(train_in[i] * w + b)
        print(f"{train_in[i]} * {w} + {b} = {res}")
    print()

w = random.randint(0, 99)
b = random.randint(0, 99)

w, b = train(1000, w, b)
print_res(w, b)

# cost
# trekk fra derivert

