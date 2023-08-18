import subprocess
from subprocess import STDOUT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
subprocess.run(['python','-m','train.train_mdm','--save_dir','save/uncond-dp','--dataset','humanact12','--cond_mask_prob','0','--lambda_rcxyz','1','--lambda_vel','1','--lambda_fc','1 ','--unconstrained','--dependent'
               ], stderr=STDOUT)