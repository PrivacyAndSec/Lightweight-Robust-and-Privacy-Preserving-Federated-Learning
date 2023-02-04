import Argument
import Run
args = Argument.compare_dataset()
for arg in args:
    Run.run(arg)