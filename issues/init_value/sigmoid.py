import numpy as np

def sigmoid(x):
    return 1/(np.exp(-x) + 1)


if __name__ == '__main__':
    print(sigmoid(-1.5))