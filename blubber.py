import sys
import numpy as np
import copy
import math
import time

class Network:
    class Gradient:
        def __init__(self):
            self._ws = []
            self._bs = []

    def __init__(self, arch: list, train_in, train_out):
        self._arch = arch

        self._train_in = train_in
        self._train_out = train_out

        self._ws = []
        self._bs = []
        self._as = []
        self._gradient = self.Gradient()

        self._h = 1e-3
        self._rate = 1e-1

        rows = train_in.shape[0]
        cols = train_in.shape[1]

        # Feed the input into the network
        self._as.append(train_in)

        for layer in range(len(self._arch)):
            n_count = self._arch[layer]
            self._ws.append(np.random.randn(n_count, rows) * 0.01)
            self._bs.append(np.random.randn(n_count, cols) * 0.01)
            self._as.append(np.zeros((n_count, cols)))

            # Create the gradient
            self._gradient._ws.append(np.zeros((n_count, rows)))
            self._gradient._bs.append(np.zeros((n_count, cols)))

            rows = len(self._as[layer + 1])

    def sigmoid(self, x: list):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        for layer in range(len(self._arch)):
            self._as[layer + 1] = self._ws[layer] @ self._as[layer] # Multiply weights
            self._as[layer + 1] += self._bs[layer] # Add the bias
            self._as[layer + 1] = self.sigmoid(self._as[layer + 1]) # Apply activation

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

    def train_fin_diff(self, iterations):
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

    # -------------------- Backpropagation --------------------
    def backprop(self):
        self.forward()

        delta_output = 2 * (self.out() - self._train_out)

        # Propagate backwards
        for layer in reversed(range(len(self._arch))):
            # Gradient for the current layer (using sigmoid)
            delta_layer = delta_output * self._as[layer + 1] * (1 - self._as[layer + 1])
            delta_layer *= 2 / len(self._train_in[0])

            # Gradients for weights and biases
            self._gradient._ws[layer] = delta_layer @ self._as[layer].T
            self._gradient._bs[layer] = delta_layer

            # Propagate the gradient to the previous layer
            delta_output = np.dot(self._ws[layer].T, delta_layer)

        # Update parameters
        for layer in range(len(self._arch)):
            self._ws[layer] -= self._gradient._ws[layer] * self._rate
            self._bs[layer] -= self._gradient._bs[layer] * self._rate

    def train(self, iterations, input, output, load=False):
        # Set the input and output data
        self._train_in = input
        self._as[0] = input
        self._train_out = output
        
        # Actual training
        for i in range(iterations):
            self.backprop()

            # Add 1 because the range function is non inclusive
            if (i / iterations * 100 + 1) % 1 == 0 and load:
                progress = math.ceil(i / iterations * 100 + 1)
                sys.stdout.write('\r')
                sys.stdout.write("[%-10s] %d%%" % ('=' * int(progress/10), progress))
                sys.stdout.flush()
        print()

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

def label_to_mat(label):
    mat = np.zeros((10, 1))
    mat[label] = 1
    return mat.T

def main():
    # Read the mnist training dataset
    images = []
    labels = []

    count = 0
    with open("mnist_train.csv") as f:
        for line in f:
            parts = line.rstrip().split(",")
            label = int(parts[0])
            pixels = np.array([int(x) for x in parts[1:]])[np.newaxis]
            
            labels.append(label)
            images.append(pixels)

            count += 1
            if count == 10000: break

    # We need the shape of the data to initialize the network
    out = label_to_mat(labels[0])
    nn = Network([16, 16, 10], images[0].T, out.T)

    for i in range(len(images)):
        nn.train(20, images[i].T, label_to_mat(labels[i]).T)
        print(f"\nOutput: {labels[i]}")
        print(nn.out())


if __name__ == "__main__":
    start_time = time.time()
    main()
    timed = time.time() - start_time
    print(f"Minutes: {timed // 60}, seconds: {timed % 60}")



# Our data was transposed, so we pick out input and output by rows instead of columns
# inp = train[0:2]
# out = train[2][np.newaxis] # Stay as matrix
