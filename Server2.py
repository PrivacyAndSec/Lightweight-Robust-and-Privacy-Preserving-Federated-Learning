import numpy as np

class S2():
    def __init__(self, N, K, D):
        self.client_number = N
        self.iterations = K
        self.dimension = D
        self.n2 = np.zeros((self.client_number,self.dimension))
        self.y = np.zeros((self.client_number, self.dimension))
        self.b = np.zeros((self.client_number, self.dimension))
        self.alpha = np.zeros(self.client_number)
        self.a = np.zeros((self.client_number, self.dimension))
        self.acos_list = []
        self.cos_list = []
        self.person_list = []

    def PRG(self):  # The n1 of epoch k
        self.n2 = np.zeros((self.client_number, self.dimension))
        for d in range(0, self.dimension):
            for i in range(0, self.client_number):
                if i != self.client_number - 1:
                    r = np.random.randint(-0xfffff, 0xffffff)
                    while r == 0:
                        r = np.random.randint(-0xfffff, 0xffffff)
                    self.n2[i][d] = r
                else:
                    self.n2[i][d] = -np.sum(self.n2[:,d])
                    while self.n2[i][d] == 0:
                        r = np.random.randint(-0xfffff, 0xffffff)
                        while r == 0:
                            r = np.random.randint(-0xfffff, 0xffffff)
                        self.n2[i-1][d] = r
                        self.n2[i][d] = -np.sum(self.n2[:, d])
            assert np.sum(self.n2[:,d]) == 0

    def receive_y(self,y_i, i):
        self.y[i] = y_i

    def calculate_alpha(self, method, last_g=None, compare_coef=None):
        if method == "ShieldFL":
            self.ShieldFL(last_g)
        elif method == "PEFL":
            self.PEFL()
        elif method == "FL":
            self.FL(compare_coef)
        elif method == "PFL":
            self.PFL(compare_coef)
        else:
            raise Exception("There is no method \"{}\".".format(method))


    def verify(self, method, last_g = None, compare_coef = None):
        self.b = self.y
        if method == "ShieldFL":
            self.calculate_alpha(method, last_g=last_g)
        else:
            self.calculate_alpha(method, compare_coef=compare_coef)
        for i in range(0, self.client_number):
            self.a[i] = self.b[i] * self.alpha[i]
        self.a = self.a
        return self.a

    #The way to find a benchmark gradient
    def m_Krum(self, m, f): #Because of n1, Euclidean distance is lose efficacy. Use similarity to measure. However, Cosine similarity is not sensitive to values, we turn to use adjusted_cosine.
        choice_list = np.zeros(m) -9999 #initialize
        mean = np.zeros(self.dimension)
        for k in range(0, m):
            distance = np.zeros((self.client_number, self.client_number))
            for i in range(0, self.client_number):
                if i in choice_list: #Skip the selected gi
                    continue
                for j in range(0, self.client_number):
                    if j in choice_list:  # Skip the selected gj
                        continue
                    if i == j:
                        continue
                    else:
                        distance[i][j] = self.adjusted_cosine(self.b[i], self.b[j])
            kr = np.zeros(self.client_number)
            d_i_min = -9999
            choice = -1
            for i in range(0, self.client_number):
                d_i = distance[i,:]
                d_i.sort()
                for j in range(0, self.client_number - f - 1):
                    kr[i] = kr[i] + d_i[-j]
                if kr[i] > d_i_min:
                    choice = i
                    d_i_min = kr[i]
            choice_list[k] = choice
            mean = mean + self.b[choice]
        mean = mean / m
        return mean

    def Trimmed_mean(self, beta): #[11]ofshieldFL
        trimmed_mean = np.zeros(self.dimension)
        for d in range(0, self.dimension):
            x = self.b[:,d]
            x.sort()
            trimmed_mean[d] = np.sum(x)
            for m in range(0, beta):
                trimmed_mean[d] = trimmed_mean[d] - x[m] - x[-m]
        return trimmed_mean

    def Median(self):
        median = np.zeros(self.dimension)
        for d in range(0, self.dimension):
            x = self.b[:,d]
            x.sort()
            if self.client_number % 2 == 0:
                median[d] = (x[int(self.client_number/2)] + x[int(self.client_number/2 - 1)]) / 2
            else:
                median[d] = x[int(self.client_number / 2)]
        return median


    #The way to calculate the distance between two gradient

    def pearson(self,x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)
        cov_x = [X - x_mean for X in x]
        cov_y = [Y - y_mean for Y in y]
        cov_xy = 0
        for i in range(0, len(cov_y)):
            cov_xy = cov_xy + cov_x[i] * cov_y[i]
        cov_xy = cov_xy / len(cov_y)
        pearson = cov_xy / (x_std * y_std)
        return pearson

    def cosine(self, x, y):
        div1 = np.sqrt(np.sum(x*x))
        div2 = np.sqrt(np.sum(y*y))
        cos = np.sum(np.multiply(x, y)) / (div1 * div2)
        return cos

    def adjusted_cosine(self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        mean = (x_mean + y_mean) / 2
        a_cos = 0
        div1 = 0
        div2 = 0
        for i in range(0, len(x)):
            a_cos = a_cos + (x[i] - mean) * (y[i] - mean)
            div1 = div1 + (x[i] - mean) * (x[i] - mean)
            div2 = div2 + (y[i] - mean) * (y[i] - mean)
        div1 = np.sqrt(div1)
        div2 = np.sqrt(div2)
        a_cos = a_cos / (div1 * div2)
        return a_cos

    #The way to calculate weight of each parameter in the aggregation

    def ShieldFL(self, last_g): # cosine
        if last_g.all() == np.zeros(self.dimension).all(): #判断是否是第一次聚合，此时默认没有用户造假。
            self.alpha = np.ones(self.client_number) / self.client_number
        else:
            coef = np.zeros(self.client_number)
            lowest_cos = 2
            benchmark = None
            for i in range(0, self.client_number):
                cos = self.cosine(self.b[i], last_g)
                if cos < lowest_cos:
                    benchmark = self.b[i]
            for i in range(0,self.client_number):
                coef[i] = 1 - self.cosine(self.b[i], benchmark)
            self.alpha = coef / np.sum(coef)

    def PEFL(self): #pearson + median
        coef = np.zeros(self.client_number)
        benchmark = self.Median()
        for i in range(0, self.client_number):
            pearson = self.pearson(self.b[i], benchmark)
            coef[i] = max(0, np.emath.log((1.1 + pearson) / (1.1 - pearson)) - 0.5)
        self.alpha = coef / np.sum(coef)

    def FL(self,compare_coef):
        self.alpha = np.ones(self.client_number) / self.client_number
        if compare_coef is True:
            benchmark = self.Median()
            pearson = self.pearson(self.b[1], benchmark)
            self.person_list.append(pearson)
            cos = self.cosine(self.b[1], benchmark)
            self.cos_list.append(cos)
            acos = self.adjusted_cosine(self.b[1], benchmark)
            self.acos_list.append(acos)

    def PFL(self,compare_coef):
        coef = np.zeros(self.client_number)
        benchmark = self.Median()
        for i in range(0, self.client_number):
            acos = self.adjusted_cosine(self.b[i], benchmark)
            if acos < 0.4:
                acos = 0
            coef[i] = np.tanh(acos)
        self.alpha = coef / np.sum(coef)

        if compare_coef is True:
            benchmark = self.Median()
            pearson = self.pearson(self.b[1], benchmark)
            self.person_list.append(pearson)
            cos = self.cosine(self.b[1], benchmark)
            self.cos_list.append(cos)
            acos = self.adjusted_cosine(self.b[1], benchmark)
            self.acos_list.append(acos)