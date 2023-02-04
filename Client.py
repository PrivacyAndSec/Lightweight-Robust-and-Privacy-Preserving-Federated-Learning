import numpy as np
import Trainer

class C():
    def __init__(self, N, id, n, m, model_name):
        self.id = id
        self.client_number = N
        self.Trianer = Trainer.Trianer(model_name)
        if n == 32:
            self.dimension = 1841162
        else:
            self.dimension = 512 * (512 + 10 + 2 + n * m) + 10
        self.n2 = np.zeros(self.dimension)
        self.n1 = np.zeros(self.dimension)
        self.g = np.zeros(self.dimension)

    def set_n1(self, n1):
        self.n1 = n1

    def set_n2(self, n2):
        self.n2 = n2

    def local_train(self, dataloader,round_id, arg, malicious):
        grads_local, loss = self.Trianer.local_update(dataloader, self.id, round_id, arg, malicious)
        self.g = self.Trianer.get_gradient(grads_local)

    def update(self, g): # Use the aggregation result to update gradient
        #获得聚合后的梯度g，用g更新上轮的系数，从而获得本轮的系数
        self.Trianer.set_gradient(g)

    def recall(self, test_dataloader, arg, source=None ):
        self.Trianer.recall(test_dataloader, arg, source)

    def score(self, test_dataloader,arg, round_id):
        loss_normal, acc_normal = self.Trianer.test_model(test_dataloader=test_dataloader, epoch=round_id, arg=arg)

    def encry(self, arg=None):
        #通过本轮系数和上轮系数，获得本轮所计算的梯度g
        if arg is None:
            y = self.g
        else:
            y = self.g * arg.scaling
            y = y
        return y


