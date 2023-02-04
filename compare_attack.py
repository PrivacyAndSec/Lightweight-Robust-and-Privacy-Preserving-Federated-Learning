import Argument
import Run
args = Argument.compare_attack()
for arg in args:
    Run.run(arg)