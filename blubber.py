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
        self.ws = []
        self.bs = []
        self.as = []

        rows = len(train_in)

        as.append(mat_gen(rows, len(train_in[0])), 0, 0)
        for i in arch
            for j in i:
                ws.append(mat_gen(rows, 1, 0, 1))
                bs.append(mat_gen(rows, 1, 0, 1))
            as.append(mat_gen(rows, i, 0, 0))

    def print(self):
        for i in range(len(as)):
            print("\nas")
            mat_print(as[i])
                        
        for i in range(len(ws)):
            print("\nw:")
            mat_print(ws[i])    
            print("\nb:")
            mat_print(bs[i])

def main():
    nn = Network([2, 3, 2, 1])
    nn.print()
    
if __name__ == "__main__":
    main()
