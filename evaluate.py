import subprocess
from subprocess import STDOUT
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# subprocess.run(['python','-m','eval.eval_humanact12_uestc','--model','./save/uncond_baseline/model000600000.pt','--eval_mode','full'], stderr=STDOUT)
subprocess.run(['python','-m','eval.eval_humanact12_uestc','--model','./save/uncond_baseline/model000450000.pt','--eval_mode','full'], stderr=STDOUT)