from matrix import *
import math

# Now our shit is sideways yo
train_in = [[0, 0, 1, 1],
            [0, 1, 0, 1]]

train_out = [[0, 1, 1, 1]]



class Network:
    def __init__(self, arch: list):
        self._arch = arch
        self._ws = []
        self._bs = []
        self._as = []
        self._gradient = None # Must be created after initialization

        rows = len(train_in)
        cols = len(train_in[0])

        self._as.append(mat_gen(rows, cols, 0, 0))
        # Insert the training data
        mat_sum(self._as[0], train_in)

        for layer in range(len(self._arch)):
            n_count = self._arch[layer]
            self._ws.append(mat_gen(n_count, rows, 0, 1))
            self._bs.append(mat_gen(n_count, cols, 0, 1))
            self._as.append(mat_gen(n_count, cols, 0, 0))
            rows = len(self._as[layer + 1])


    # Must be called after creating a network
    def init(self):
        self._gradient = Network(self._arch)


    def sigmoid(self, x):
        for rows in range(len(x)):
            for cols in range(len(x[rows])):
                x[rows][cols]  = 1 / (1 + math.exp(-x[rows][cols]))


    def forward(self):
        for layer in range(len(self._arch)):
            self._as[layer + 1] = mat_dot(self._ws[layer], self._as[layer])
            mat_sum(self._as[layer + 1], self._bs[layer])
            self.sigmoid(self._as[layer + 1])


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
    nn = Network([5, 1, 1])
    nn.init()
    # nn.print()
    print("\nForwarding")
    nn.forward()
    # nn.print()
    mat_print(nn.out())


if __name__ == "__main__":
    main()

