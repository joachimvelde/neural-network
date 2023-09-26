from matrix import *

train_in = [[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]]

train_out = [[0],
             [1],
             [1],
             [1]]



class Network:
    def __init__(self, arch: list):
        self._arch = arch
        self._ws = "yo"
        self._bs = []
        self._as = []
        self._gradient = None # Must be created after initialization

    # Must be called after creating a network
    def init(self):
        self.gradient = Network(self._arch)


def main():
    nn = Network([2, 1])
    nn.init()

    print(len(nn._gradient._ws))

    print("oh")

if __name__ == "__main__":
    main()

