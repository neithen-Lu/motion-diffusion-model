import subprocess
from subprocess import STDOUT
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# subprocess.run(['python','-m','eval.eval_humanact12_uestc','--model','./save/uncond_baseline/model000600000.pt','--eval_mode','full'], stderr=STDOUT)
# subprocess.run(['python','-m','eval.eval_humanact12_uestc','--model','/home/qindafei/KX/motion-diffusion-model/save/a2m-uestc-dp/model000100000.pt','--eval_mode','debug'], stderr=STDOUT)
# subprocess.run(['python','-m','eval.eval_humanact12_uestc','--model','/home/qindafei/KX/motion-diffusion-model/save/a2m-uestc-dp/model000100000.pt','--eval_mode','debug'], stderr=STDOUT)
# subprocess.run(['python','-m','eval.eval_humanact12_uestc','--model','/home/qindafei/KX/motion-diffusion-model/save/a2m-humanact12-base/model000450000.pt','--eval_mode','full'], stderr=STDOUT)
subprocess.run(['python','-m','eval.eval_humanact12_uestc','--model','/home/qindafei/KX/motion-diffusion-model/save/a2m-uestc-dp/model000550000.pt','--eval_mode','full'], stderr=STDOUT)