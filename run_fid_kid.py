import subprocess
from subprocess import STDOUT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# subprocess.run([
#     'python', 'fid_kid.py',
#     '/home/qindafei/KX/data/cifar',
#     '/home/qindafei/KX/image_diffusion/result/local2_decay0.1_128/samples_0_10000x32x32x3.npz',
#     '--num_samples', '10000',
#     '--batch_size', '100'
# ], stderr=STDOUT)
subprocess.run([
    'python', 'fid_kid.py',
    '/home/qindafei/KX/data/imagenet_train_64x64/train_64x64',
    '/home/qindafei/KX/image_diffusion/result/imagenet_base_learnsigTrue/samples_0_10000x64x64x3.npz',
    '--num_samples', '10000',
    '--batch_size', '100'
], stderr=STDOUT)