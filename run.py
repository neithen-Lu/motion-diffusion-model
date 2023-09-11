import subprocess
from subprocess import STDOUT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
subprocess.run(['python','-m','train.train_mdm','--save_dir','save/uncond-dp-005','--dataset','humanact12','--cond_mask_prob','0','--lambda_rcxyz','1','--lambda_vel','1','--lambda_fc','1 ','--unconstrained','--dependent','--decay_rate','0.05','--eval_during_training','--overwrite'
               ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/a2m-uestc-base','--dataset','uestc','--cond_mask_prob','0','--lambda_rcxyz','1','--lambda_vel','1','--lambda_fc','1 ','--eval_during_training','--overwrite'
#                ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/a2m-uestc-dp','--dataset','uestc','--cond_mask_prob','0','--lambda_rcxyz','1','--lambda_vel','1','--lambda_fc','1 ','--eval_during_training','--dependent','--overwrite'
#                ], stderr=STDOUT)

# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/humanml-ar-resume','--dataset','humanml','--eval_during_training','--dependent','--window_size','49','--ar_sample','--overwrite','--resume_checkpoint','/raid/HKU_TK_GROUP/qindafei/neithen/motion-diffusion-model/save/humanml-ar/model000300000.pt'
#                ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/humanml-ar','--dataset','humanml','--eval_during_training','--dependent','--window_size','49','--ar_sample','--overwrite'
#                ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/humanml-dp-losssig-2','--dataset','humanml','--eval_during_training','--dependent','--window_size','49','--loss_sig','--overwrite'
#                ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/kit-base','--dataset','kit','--eval_during_training'
#                ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/kit-dp','--dataset','kit','--eval_during_training','--dependent','--window_size','49','--eval_during_training','--overwrite'
#                ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/debug','--dataset','humanml','--eval_during_training','--dependent','--window_size','28','--overwrite'
#                ], stderr=STDOUT)
# subprocess.run(['python','-m','train.train_mdm','--save_dir','save/humanml-base-resume','--dataset','humanml','--eval_during_training',
#                ], stderr=STDOUT)