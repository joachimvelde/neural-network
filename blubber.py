import sys
import numpy as np
import copy
import math
import time
import random

# Print numpy arrays in a readable format
np.set_printoptions(edgeitems=30, linewidth=100, 
    formatter=dict(float=lambda x: "%.3g" % x))

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
        self._rate = 1e-4

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

    def train(self, input, output):
        # Set the input and output data
        self._train_in = input
        self._as[0] = input
        self._train_out = output
        
        self.backprop()

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

    def save(self, filename):
        weight_dict = {f"array{i}": array for i, array in enumerate(self._ws)}
        bias_dict = {f"array{i}": array for i, array in enumerate(self._bs)}

        np.savez(filename + "_w", **weight_dict)
        np.savez(filename + "_b", **bias_dict)

    def load(self, filename):
        loaded_ws = np.load(filename + "_w.npz")
        loaded_bs = np.load(filename + "_b.npz")

        layer = 0
        for key in loaded_ws.keys():
            self._ws[layer] = loaded_ws[key]
            layer += 1

        layer = 0
        for key in loaded_bs.keys():
            self._bs[layer] = loaded_bs[key]
            layer += 1

def label_to_mat(label):
    mat = np.zeros((10, 1))
    mat[label] = 1
    return mat

def compress(image):
    src_dim = 520
    dst_dim = 28
    scale = src_dim / dst_dim

    compressed = [0 for _ in range(dst_dim**2)]

    for y in range(dst_dim):
        for x in range(dst_dim):
            og_x = int(x * scale)
            og_y = int(y * scale)

            avg_colour = 0
            count = 0
            for dy in range(int(scale)):
                for dx in range(int(scale)):
                    index = (og_y * dy) * src_dim + (og_x + dx)
                    print(index)
                    if index < src_dim**2:
                        pixel_val = image[0][index]
                        avg_colour += pixel_val
                        count += 1

            if (y * dst_dim + x) < dst_dim**2:
                compressed[y * dst_dim + x] = avg_colour / count

    return np.array(compressed)[np.newaxis]

# Lower training rate
# Smaller dataset
# Less training iterations
# More epochs

# More data -> more epochs and lower training rate???

filename = "params/network"

def main():
    # Read the mnist training dataset
    images = []
    labels = []

    epochs = 20 # Careful of overfitting

    data_count = 1015 # How many images to load
    train_count = data_count - 15 # How many images to train the network on

    count = 0
    with open("mnist_train.csv") as f:
        for line in f:
            parts = line.rstrip().split(",")
            label = int(parts[0])
            pixels = np.array([int(x) for x in parts[1:]])[np.newaxis]
            
            labels.append(label)
            images.append(pixels)

            count += 1
            if count == data_count: break

    # We need the shape of the data to initialize the network
    out = label_to_mat(labels[0])
    nn = Network([128, 10], images[0].T, out)

    # For every epoch - train the network on the selected data
    for j in range(epochs):
        for i in range(train_count):
            nn.train(images[i].T, label_to_mat(labels[i]))
            nn.forward()

        # Print a loading bar to show the training progress
        if (j / epochs * 100 + 1) % 1 == 0:
            progress = math.ceil(j / epochs * 100 + 1)
            sys.stdout.write('\r')
            sys.stdout.write("[%-10s] %d%%" % ('=' * int(progress/10), progress))
            sys.stdout.flush()
    print()

    print("\n ----------TEST DATA----------")

    # Test the network on the loaded images it was not trained on
    for i in range(train_count, data_count):
        nn._as[0] = images[i].T
        nn.forward()
        print(f"\nDesired: {labels[i]}, index: {i}")
        print(nn.out())

    nn.save(filename)
    print("Saving parameters")

def load():
    # Read the mnist training dataset
    images = []
    labels = []

    # data_count = 0
    # with open("mnist_train.csv") as f:
    #     for line in f:
    #         parts = line.rstrip().split(",")
    #         label = int(parts[0])
    #         pixels = np.array([int(x) for x in parts[1:]])[np.newaxis]
    #         
    #         labels.append(label)
    #         images.append(pixels)

    #         data_count += 1

    data_count = 1
    with open("canvas_dataset.csv") as f:
        for line in f:
            parts = line.rstrip().split(",")
            label = 2
            pixels = np.array([int(x) for x in parts])[np.newaxis]

            labels.append(label)
            images.append(pixels)

    # Compress this mafakka
    print(images[0])
    # images[0] = compress(images[0])
    print(np.rint(images[0].reshape(28, 28)))

    # We need the shape of the data to initialize the network
    out = label_to_mat(labels[0])
    nn = Network([128, 10], images[0].T, out)

    nn.load(filename)
    print("Loaded parameters")

    errors = 0
    # Test the network on the whole dataset
    for i in range(data_count):
        nn._as[0] = images[i].T
        nn.forward()
        guess = np.argmax(nn.out())
        if guess != labels[i]: errors += 1

        print(f"\nDesired: {labels[i]}, index: {i}, guess: {guess}")
        print(nn.out())

    accuracy = 100 - (errors / data_count * 100)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    start_time = time.time()
    # main()
    load()
    timed = time.time() - start_time
    print(f"Minutes: {timed // 60}, seconds: {timed % 60}")

