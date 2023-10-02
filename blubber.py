import numpy as np
import copy
import math

# Use .T instead
train = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]]).transpose()

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

        # Feed the input into the network
        self._as.append(train_in)

        for layer in range(len(self._arch)):
            n_count = self._arch[layer]
            self._ws.append(np.random.randn(n_count, rows) * 1e-2)
            self._bs.append(np.random.randn(n_count, cols) * 1e-2)
            self._as.append(np.zeros((n_count, cols)))
            rows = len(self._as[layer + 1])

    def init(self):
        self._gradient = Network(self._arch, self._train_in, self._train_out)

    def sigmoid(self, x: list):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        for layer in range(len(self._arch)):
            self._as[layer + 1] = self._ws[layer] @ self._as[layer] # Multiply weights
            self._as[layer + 1] += self._bs[layer] # Add the bias
            self.sigmoid(self._as[layer + 1]) # Apply activation

    # ---------- Gradient throught finite difference ----------
    def out(self):
        return self._as[len(self._arch)]

    def cost(self):
        self.forward()
        cum_sum = 0
        for i in range(len(self.out()[0])):
            y = self.out()[0][i]
            cum_sum += (y - self._train_out[0][i])**2
        return cum_sum / len(self._train_out[0])

    def diff(self):
        # Differentiate all weights and biases
        for layer in range(len(self._arch)):
            for row in range(len(self._ws[layer])):
                # Weights
                for col in range(len(self._ws[layer][row])):
                    # Calculate finite difference
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
                    self._gradient._bs[layer][row][col] = (inc_cost - cur_cost) / self._h

    def train(self, iterations):
        for i in range(iterations):
            self.diff() # Calculate the gradient
            # Update parameters
            for layer in range(len(self._arch)):
                # Weights
                for row in range(len(self._ws[layer])):
                    for col in range(len(self._ws[layer][row])):
                        self._ws[layer][row][col] -= self._gradient._ws[layer][row][col] * self._rate
                    # Biases (will only have one column, but this is more flexible?)
                    for col in range(len(self._bs[layer][row])):
                        self._bs[layer][row][col] -= self._gradient._bs[layer][row][col] * self._rate
    # -------------------------------------------------------

    # ---------- Implementing backpropagation ----------
    def backprop(self):
        self.forward()

        delta_output = 2 * (self.out() - self._train_out)

        # Propagate backwards
        for layer in reversed(range(len(self._arch))):
            # Gradient for the current layer
            delta_layer = delta_output * self._as[layer + 1] * (1 - self._as[layer + 1])
            delta_layer *= 2 / len(self._train_in[0])

            # Gradients for weights and biases
            self._gradient._ws[layer] = delta_layer @ self._as[layer].T
            self._gradient._bs[layer] = np.sum(delta_layer, axis=1, keepdims=True)

            # Propagate the gradient to the previous layer
            delta_output = np.dot(self._ws[layer].T, delta_layer)

        # Update parameters
        for layer in range(len(self._arch)):
            self._ws[layer] -= self._gradient._ws[layer] * self._rate
            self._bs[layer] -= self._gradient._bs[layer] * self._rate


    def print_out(self):
        print(self._as[len(self._arch)].transpose())

    def print_in(self):
        print(self._as[0].transpose())

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
    # Our data was transposed, so we pick out input and output by rows instead of columns
    inp = train[0:2]
    out = train[2][np.newaxis] # Stay as matrix

    nn = Network([2, 1], inp, out)
    nn.init()

    # Training throught finite differences
    # nn.train(10000)

    nn._gradient.print()
    for i in range(10):
        nn.backprop()
    print()
    nn._gradient.print()


    print("\nInput")
    nn.print_in()

    print("\nOutput")
    nn.print_out()


if __name__ == "__main__":
    main()
