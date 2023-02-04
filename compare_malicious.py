import Argument
import Run
args = Argument.compare_malicious()
for arg in args:
    Run.run(arg)