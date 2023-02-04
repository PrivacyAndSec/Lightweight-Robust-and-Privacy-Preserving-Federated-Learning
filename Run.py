import numpy as np
import Server1
import Server2
import Client
import get_data
import random
import time
import os
def run(arg):
    if os.path.exists(arg.result_path):
        os.remove(arg.result_path)
    file = open(arg.result_path, "a+")
    if arg.attack == 'no':
        line = "clients number: {}, iteration: {}, defense: {}, dataset: {}, attack: {}\n "\
            .format(arg.N, arg.K, arg.method, arg.data_set, arg.attack)
    elif arg.attack == 'target':
        line = "clients number: {}, iteration: {}, source: {}, target: {}, defense: {}, dataset: {}, attack: {}, " \
               "malicious number: {}\n".format(arg.N, arg.K, arg.source, arg.target, arg.method, arg.data_set,
                                                  arg.attack, arg.malicious_number)
    elif arg.attack == 'untarget':
        line = "clients number: {}, iteration: {}, defense: {}, dataset: {}, attack: {}, malicious number: {}\n".format(arg.N, arg.K, arg.method, arg.data_set,
                                                  arg.attack, arg.malicious_number)
    elif arg.attack == 'backdoor':
        line = "clients number: {}, iteration: {}, target: {}, defense: {}, dataset: {}, attack: {}, malicious number: {}\n".format(arg.N, arg.K, arg.target, arg.method, arg.data_set,
                                                  arg.attack, arg.malicious_number)
    elif arg.attack == 'scaling':
        line = "clients number: {}, iteration: {}, scaling: {}, defense: {}, dataset: {}, attack: {}, malicious number: {}\n".format(arg.N, arg.K, arg.scaling, arg.method, arg.data_set,
                                                  arg.attack, arg.malicious_number)
    else:
        raise Exception("There is no such attack: {}.".format(arg.attack))
    file.write(line)
    print(line)

    file.close()
    if arg.malicious_number >= arg.N / 2:
        raise Exception("Malicious users account({}) is more than 50% of the total users({}).".format(arg.malicious_number, arg.N))
    if arg.compare_coef == True:
        malicious_id = [0,1,2,3,4,5,6,7,8]
    else:
        malicious_id = random.sample(range(0,arg.N),arg.malicious_number)
    ds = get_data.DatasetSource(arg.data_set) # get data set
    ci_dataloader = ds.get_train_dataloader(arg=arg) # load train data
    test_dataloader = ds.get_test_dataloader() # load test data


    client_time = 0
    s1_time = 0
    s2_time = 0


    C = []

    time_begin = time.time_ns()
    for i in range(0, arg.N):
        C.append(Client.C(arg.N, i, ds.n, ds.m, "CNNMNIST"))
    time_end = time.time_ns()

    client_initial_time = (time_end - time_begin) / arg.N /1000000  #ms

    time_begin = time.time_ns()
    S1 = Server1.S1(arg.N, arg.K, C[0].dimension)
    time_end = time.time_ns()
    s1_initial_time = (time_end - time_begin) /1000000  #ms

    time_begin = time.time_ns()
    S2 = Server2.S2(arg.N, arg.K, C[0].dimension)
    time_end = time.time_ns()
    s2_initial_time = (time_end - time_begin) / 1000000  # ms


    g = np.zeros(C[0].dimension) #初始化 梯度
    for k in range(0, arg.K):# Sends local gradient with two-masks
        #S1.PRG()
        #S2.PRG()

        for i in range(0,arg.N):
            time_begin = time.time_ns()
            #C[i].set_n1(S1.n1.copy())
            #C[i].set_n2(S2.n2[i].copy())
            time_end = time.time_ns()
            client_time = client_time + (time_end - time_begin) /1000000

            if i in malicious_id:
                C[i].local_train(ci_dataloader[i], k, arg, malicious=True)
            else:
                C[i].local_train(ci_dataloader[i], k, arg, malicious=False)

            if arg.attack == "target":
                C[i].recall(test_dataloader, arg, source=arg.source)
            if arg.attack == "backdoor":
                C[i].recall(test_dataloader, arg, source=arg.target)

            time_begin = time.time_ns()
            if i in malicious_id and arg.attack == "scaling":
                y_i = C[i].encry(arg)
            else:
                y_i = C[i].encry()
            S2.receive_y(y_i, i)
            time_end = time.time_ns()
            client_time = client_time + (time_end - time_begin) /1000000


    #S2 verify whether malicious users exist

        time_begin = time.time_ns()
        if arg.method == "ShieldFL":
            a = S2.verify(arg.method, g)
        else:
            a = S2.verify(arg.method, compare_coef=arg.compare_coef)
        time_end = time.time_ns()
        s2_time = s2_time + (time_end - time_begin) /1000000

        time_begin = time.time_ns()
        g = S1.aggregation(a)
        for i in range(0,arg.N):
            C[i].update(g)
        time_end = time.time_ns()
        s1_time = s1_time + (time_end - time_begin) /1000000
        if (k + 1) % 2 == 0 :
            file = open(arg.result_path, "a+")
            file.write("Global model result at epoch {}-th\n".format(k))
            file.close()
            print("Global model result at epoch {}-th".format(k))
            C[0].score(test_dataloader,arg, k)
            C[0].recall(test_dataloader, arg)

        file = open(arg.result_path, "a+")
        file.close()

    file = open(arg.result_path, "a+")
    file.write("----------client initial time：{}ms----------\n".format(round(client_initial_time)))
    file.write("--------------s1 initial time：{}ms----------\n".format(round(s1_initial_time)))
    file.write("--------------s2 initial time：{}ms----------\n".format(round(s2_initial_time)))
    file.write("------------------client time：{}ms----------\n".format(round(client_time)))
    file.write("------------------s1 time：{}ms----------\n".format(round(s1_time)))
    file.write("------------------s2 time：{}ms----------\n".format(round(s2_time)))
    file.close()

    if arg.compare_coef == True:
        file = open("{}-{}.txt".format(arg.method,arg.attack), "a+")
        file.write("acos\n")
        for acos in S2.acos_list:
            file.write("{},".format(acos))
        file.write("\ncos\n")
        for cos in S2.cos_list:
            file.write("{},".format(cos))
        file.write("\nper\n")
        for per in S2.person_list:
            file.write("{},".format(per))
        file.close()









