import matplotlib.pyplot as plt
import numpy as np  # 导入numpy库
from mpl_toolkits import mplot3d

def coef3d():
    attacks = ['s','t','u']
    methods = ['FL', 'PFL']
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for method in methods:
        for attack in attacks:
            file_name = "result/compare-coefficient/{}/{}-acos.txt".format(method,attack)  # 定义数据文件
            acos_data = np.loadtxt(file_name, dtype='float32', delimiter=',')  # 获取数据

            file_name = "result/compare-coefficient/{}/{}-cos.txt".format(method,attack)  # 定义数据文件
            cos_data = np.loadtxt(file_name, dtype='float32', delimiter=',')  # 获取数据

            file_name = "result/compare-coefficient/{}/{}-per.txt".format(method,attack)  # 定义数据文件
            per_data = np.loadtxt(file_name, dtype='float32', delimiter=',')  # 获取数据

            ax.plot3D(np.arange(100), np.ones(100), acos_data, 'r', label='Adjusted Cosine Similarity')
            ax.plot3D(np.arange(100), 2*np.ones(100), cos_data, 'g', label='Cosine Similarity')
            ax.plot3D(np.arange(100), 3*np.ones(100), per_data, 'b', label='Perason Coefficient')

            ax.set_xlabel("epoch")
            ax.set_ylabel("method")
            ax.set_zlabel("coefficient")

            plt.show()


def coefficient():
    attacks = ['s','t','u']
    methods = ['FL', 'PFL']
    for method in methods:
        for attack in attacks:
            file_name = "result/compare-coefficient/{}/{}-acos.txt".format(method,attack)  # 定义数据文件
            acos_data = np.loadtxt(file_name, dtype='float32', delimiter=',')  # 获取数据

            file_name = "result/compare-coefficient/{}/{}-cos.txt".format(method,attack)  # 定义数据文件
            cos_data = np.loadtxt(file_name, dtype='float32', delimiter=',')  # 获取数据

            file_name = "result/compare-coefficient/{}/{}-per.txt".format(method,attack)  # 定义数据文件
            per_data = np.loadtxt(file_name, dtype='float32', delimiter=',')  # 获取数据

            x = np.arange(100)
            plt.xlabel("epoch")
            plt.plot(x, acos_data, color='r', linestyle='--', label='Adjusted Cosine Similarity')
            plt.plot(x, cos_data, color='g', linestyle='-.', label='Cosine Similarity')
            plt.plot(x, per_data, color="b", label='Perason Coefficient')
            plt.legend()
            plt.savefig('./picture/{}-{}.png'.format(method,attack))
            plt.show()

def malicious():
    dataset = "cifar10"
    x = [0, 10 , 20, 30, 40, 49]

    if dataset == "mnist":

        tACC = [97.39,97.38,97.35,97.21,97.11,96.66]
        SACC = [98.61,98.59,98.50,97.62,96.65,94.01]
        sACC = [97.39,97.18,97.25,97.10,97.33,97.31]
        uACC = [97.39,97.38,97.24,97.06,96.84,96.37]
    elif dataset == "fmnist":
        tACC = [86.72, 86.82, 87.04, 86.64, 86.39, 86.28]
        SACC = [96.11, 96.10, 95.30, 94.40, 91.00, 89.90]
        sACC = [86.72, 86.76, 86.91, 86.74, 87.11, 87.08]
        uACC = [86.72, 86.61, 86.78, 86.16, 85.21, 83.92]
    else:
        tACC = [52.78, 52.26, 51.83, 51.27, 50.33, 49.15]
        SACC = [62.20, 57.98, 50.38, 41.28, 32.67, 19.27]
        sACC = [52.78, 53.65, 51.98, 52.36, 53.97, 54.26]
        uACC = [52.78, 51.34, 48.67, 46.34, 44.77, 42.79]

    plt.xlabel("Percent of malicious users (%)")
    plt.ylabel("Accuracy (%)")
    plt.plot(x, tACC, color='r', linestyle='--', label = 'target-ACC')
    plt.plot(x, SACC, color='g', linestyle='-.', label = 'target-SACC')
    plt.plot(x, uACC, color="b", marker='+', label = 'untarget-ACC')
    plt.plot(x, sACC, color='m', marker='x', label = 'scaling-ACC')
    plt.legend()
    plt.savefig('./picture/{}-Percent.png'.format(dataset))
    plt.show()

#coefficient()
coef3d()