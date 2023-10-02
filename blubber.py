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

        self._h = 1e-3
        self._rate = 1e-1

        rows = train_in.shape[0]
        cols = train_in.shape[1]

        # Feed the input into the network (probably dont need to deepcopy)
        self._as.append(copy.deepcopy(train_in))

        for layer in range(len(self._arch)):
            n_count = self._arch[layer]
            self._ws.append(np.random.rand(n_count, rows))
            self._bs.append(np.random.rand(n_count, cols))
            self._as.append(np.zeros((n_count, cols)))
            rows = len(self._as[layer + 1])

    def init(self):
        self._gradient = Network(self._arch, self._train_in, self._train_out)
        print("The gradient was initialised")
        self._gradient.print()
        print("End gradient")

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
        for layer in range(len(self._arch)):
            # Weights
            print("Iterating over this weight matrix:")
            print(self._ws[layer])
            print(self._gradient._ws[layer])
            print("---")
            for row in range(len(self._ws[layer])):
                for col in range(len(self._ws[layer][row])):
                    # Calculate finite difference
                    print(f"Layer: {layer}, row: {row}, col: {col}")
                    self._ws[layer][row][col] += self._h
                    inc_cost = self.cost()
                    self._ws[layer][row][col] -= self._h
                    cur_cost = self.cost()
                    self._gradient._ws[layer][row][col] = (inc_cost - cur_cost) / self._h
                # Biases
                for col in range(len(self._bs[layer][row])):
                    self._bs[layer][row][col] += self._h
                    inc_cost = self.cost()
                    self._bs[layer][row][col] -= self._h
                    cur_cost = self.cost()
                    self._gradient._ws[layer][row][col] = (inc_cost - cur_cost) / self._h

    def train(self):
        self.forward()
        self.diff() # Calculate the gradient
        # Update parameters
        #  for layer in range(len(self._arch)):
        #      # Weights
        #      for row in range(len(self._ws[layer])):
        #          for col in range(len(self._ws[layer][row])):
        #              self._gradient._ws[layer][row][col] -= self._wd[layer][row][col] * self._rate
        #          # Biases (will only have one column, but this is more flexible?)
        #          for col in range(len(self._bs[layer][row])):
        #              self._gradient._ws[layer][row][col] -= self._bd[layer][row][col] * self._rate
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
    nn.init()

    print("network")
    nn.print_out()
    print("end network")

    nn.train()

    # print()
    # nn.print_out()
    # print(f"cost = {nn.cost()}")


if __name__ == "__main__":
    main()
