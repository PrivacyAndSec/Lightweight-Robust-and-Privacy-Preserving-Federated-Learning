import numpy as np

class S1():
    def __init__(self, N, K, D): #Initialize the S1
        self.client_number = N
        self.iterations = K
        self.dimension = D
        self.n1 = np.zeros(self.dimension)
        self.y = np.zeros((self.client_number, self.dimension))
        self.W = np.zeros(self.dimension)

    def PRG(self): #The n1 of epoch k
        self.n1 = np.zeros(self.dimension)
        for d in range(0, self.dimension):
            r = np.random.randint(-0xfffff, 0xffffff)
            while r == 0:
                r = np.random.randint(-0xfffff, 0xffffff)
            self.n1[d] = r

    def receive_y(self,y_i, i):
        self.y[i] = y_i

    def aggregation(self, a):
        g = np.zeros(self.dimension)
        for i in range(0, self.client_number):
            g = g + a[i]
        return g