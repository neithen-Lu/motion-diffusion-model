import subprocess
from subprocess import STDOUT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/uncond-dp','--dataset','humanact12','--cond_mask_prob','0','--lambda_rcxyz','1','--lambda_vel','1','--lambda_fc','1 ','--unconstrained','--dependent'
#                ], stderr=STDOUT)
subprocess.run(['python','-m','train.train_mdm','--save_dir','save/a2m-humanact12-base','--dataset','humanact12','--cond_mask_prob','0','--lambda_rcxyz','1','--lambda_vel','1','--lambda_fc','1 ','--eval_during_training'
               ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/humanml_dp','--dataset','humanml','--dependent','--eval_during_training'
#                ], stderr=STDOUT)