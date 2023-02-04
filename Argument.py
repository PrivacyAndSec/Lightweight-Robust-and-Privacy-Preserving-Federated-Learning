
class arguments:
    def __init__(self):
        self.name = ""
        self.N = 20
        self.K = 100
        self.source = 1
        self.target = 4
        self.label_number = 10
        self.method = "PFL" #methods = ["PEFL", "FL", "ShieldFL, Pair-mask"]
        self.data_set = 'mnist' #Backdoor attacks 使用 cifar10
        self.attack = 'target'  # attacks = ["target", 'untarget',  'backdoor', 'no', 'scaling']
        self.malicious_number = 9
        self.scaling = 1
        self.result_path = 'result/.txt'.format(self.method, self.attack)
        self.compare_coef = False



def compare_attack():
    methods = ['PEFL', 'ShieldFL', 'PFL']
    methods = ['PFL']
    args = []

    for method in methods:
        """
        arg_basline = arguments()
        arg_basline.name = "baseline"
        arg_basline.attack = "no"
        arg_basline.method = method
        arg_basline.result_path = "result/compare-attack-{}/no/{}.txt".format(arg_basline.data_set,method)
        args.append(arg_basline)
        """
        arg_scaling = arguments()
        arg_scaling.name = "scaling attack"
        arg_scaling.attack = "scaling"
        arg_scaling.scaling = 999
        arg_scaling.method = method
        arg_scaling.result_path = "result/compare-attack-{}/scaling/{}1.txt".format(arg_scaling.data_set,method)
        arg_scaling.compare_coef = True
        args.append(arg_scaling)


        arg_target = arguments()
        arg_target.name = "target attack(flapping attack)"
        arg_target.attack = "target"
        arg_target.method = method
        arg_target.result_path = "result/compare-attack-{}/target/{}1.txt".format(arg_target.data_set,method)
        arg_target.compare_coef = True
        args.append(arg_target)

        arg_untarget = arguments()
        arg_untarget.name = "untarget attack"
        arg_untarget.attack = "untarget"
        arg_untarget.method = method
        arg_untarget.result_path = "result/compare-attack-{}/untarget/{}1.txt".format(arg_untarget.data_set,method)
        arg_untarget.compare_coef = True
        args.append(arg_untarget)
        """
        
        arg_backdoor = arguments()
        arg_backdoor.name = "backdoor attack"
        arg_backdoor.attack = "backdoor"
        arg_backdoor.method = method
        arg_backdoor.result_path = "result/compare-attack-{}/backdoor/{}.txt".format(arg_backdoor.data_set,method)
        args.append(arg_backdoor)
        
        """



    return args

def compare_malicious():
    """
    percent = 0
    arg_basline = arguments()
    arg_basline.name = "baseline 0%malicious"
    arg_basline.malicious_number = int(arg_basline.N * percent)
    arg_basline.result_path = "result/compare-malicious/0%.txt"
    """
    percent = 0.1
    arg_1 = arguments()
    arg_1.name = "10%malicious"
    arg_1.malicious_number = int(arg_1.N * percent)
    arg_1.result_path = "result/compare-malicious/t10%.txt"

    percent = 0.2
    arg_2 = arguments()
    arg_2.name = "20%malicious"
    arg_2.malicious_number = int(arg_2.N * percent)
    arg_2.result_path = "result/compare-malicious/t20%.txt"

    percent = 0.3
    arg_3 = arguments()
    arg_3.name = "30%malicious"
    arg_3.malicious_number = int(arg_3.N * percent)
    arg_3.result_path = "result/compare-malicious/t30%.txt"

    percent = 0.4
    arg_4 = arguments()
    arg_4.name = "40%malicious"
    arg_4.malicious_number = int(arg_4.N * percent)
    arg_4.result_path = "result/compare-malicious/t40%.txt"

    percent = 0.49
    arg_5 = arguments()
    arg_5.name = "49%malicious"
    arg_5.malicious_number = int(arg_5.N * percent)
    arg_5.result_path = "result/compare-malicious/t49%.txt"

    return arg_1, arg_2, arg_3, arg_4, arg_5









