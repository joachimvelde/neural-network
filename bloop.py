import numpy as np
import copy
import math

train = np.array([[0, 0],
                  [1, 2],
                  [2, 4],
                  [3, 6]]).transpose()

class Network:
    def __init__(self, arch: list, train_in, train_out):
        self._arch = arch

        self._train_in = train_in
        self._train_out = train_out

        self._ws = []
        self._bs = []
        self._as = []
        self._gradient = None

        rows = train_in.shape[0]
        cols = train_in.shape[1]

        # Feed the input into the network
        self._as.append(copy.deepcopy(train_in))

        for layer in range(len(self._arch)):
            n_count = self._arch[layer]
            self._ws.append(np.random.rand(n_count, rows))
            self._bs.append(np.random.rand(n_count, cols))
            self._as.append(np.zeros((n_count, cols)))
            rows = len(self._as[layer + 1])

    def sigmoid(self, x: list):
        for row in range(len(x)):
            for col in range(len(x[row])):
                x[row][col] = 1 / (1 + math.exp(x[row][col]))

    def forward(self):
        for layer in range(len(self._arch)):
            self._as[layer + 1] = self._ws[layer] @ self._as[layer] # Multiply weights
            self._as[layer + 1] += self._bs[layer] # Add the bias
            self.sigmoid(self._as[layer + 1]) # Apply activation

    # ---------- Gradient throught finite difference ----------
    def out(self):
        return self._as[len(self._arch)]

    def cost(self):
        cum_sum = 0
        for i in range(len(self.out()[0])):
            y = self.out()[0][i]
            cum_sum += (y - self._train_out[0][i])**2
        return cum_sum / len(self._train_out[0])

    def diff(self):
        # Differentiate all weights and biases
        None

    def train(self):
        for layer in range(len(self._arch)):
            self._ # Trenger å definere en gradient
    # -------------------------------------------------------

    def print_out(self):
        print(self._as[len(self._arch)].transpose())

    def print(self):
        for i in range(len(self._as)):
            print("\nas")
            print(self._as[i])

        for i in range(len(self._ws)):
            print("\nw:")
            print(self._ws[i])    
            print("\nb:")
            print(self._bs[i])

def main():
    inp = train[0][np.newaxis] # Stay as matrix
    out = train[1][np.newaxis]

    nn = Network([2, 1], inp, out)
    nn.print_out()

    nn.forward()
    print()
    nn.print_out()
    print(f"cost = {nn.cost()}")


if __name__ == "__main__":
    main()
